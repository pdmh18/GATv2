import json
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

torch.manual_seed(42) # cho phép random của pytorch
np.random.seed(42) # random cho thư viện NumPy (thư viện Python chuyên về tính toán số học với mảng số)

DATA_PATH = "synthetic_banking_data.json"

# 1. Đọc dữ liệu, tách lại thành danh sách account (node) và transaction (edge)
with open(DATA_PATH, encoding="utf-8") as f:
    accounts_raw = json.load(f) # đọc filw json

all_txns = []
for acc in accounts_raw:
    all_txns.extend(acc["transactions"]) # tách traansacsion vào 1 mảng riêng là all_txns để có j train để build

print(f"Số tài khoản: {len(accounts_raw)}, số giao dịch: {len(all_txns)}")

# 2. Tính node feature: CHỈ dùng thống kê hành vi TRƯỚC 1 MỐC THỜI GIAN CỐ ĐỊNH
#    (feature cutoff) để tránh leakage - không được dùng giao dịch xảy ra
#    SAU mốc này (bao gồm cả chính giao dịch gian lận cần dự đoán) để tính feature.
#    KYC giữ nguyên vì đây là thông tin có ngay từ đầu, không phải thống kê
#    tổng hợp từ giao dịch nên không gây leakage.

account_index = {acc["account_id"]: i for i, acc in enumerate(accounts_raw)} # tạo bảng để tra cứu vị trí tk, enumerate(accounts_raw) lấy chỉ số (0, 1, 2...) vừa lấy nội dung từng tài khoản khi lặp qua danh sách
n_accounts = len(accounts_raw) #Kết quả account_index là 1 dictionary dạng: {"ACC_000000": 0, "ACC_000001": 1, ...}

SIM_START = min(datetime.fromisoformat(t["created_at"]) for t in all_txns) # tìm thời điểm sớm nhất trog bộ dữ liệu

def day_of(txn): # tính xem nó giao dịch vào ngày thứ mất kể từ ngày SIM_START
    return (datetime.fromisoformat(txn["created_at"]) - SIM_START).days #ví dụ SIM_START = 1/3/2026, và 1 giao dịch có created_at = 5/3/2026 => day_of() trả về 4 (cách 4 ngày).


TRAIN_END_DAY = 20   # ngày 0-19: train
VAL_END_DAY = 25     # ngày 20-24: validation, ngày 25-29: test
FEATURE_CUTOFF_DAY = TRAIN_END_DAY  # node feature chỉ dùng dữ liệu TRƯỚC mốc này

# Gom giao dịch theo tài khoản nguồn, CHỈ LẤY giao dịch xảy ra TRƯỚC mốc cutoff
txns_by_source_before_cutoff = defaultdict(list)
for t in all_txns: # duyệt all_txns
    if day_of(t) < FEATURE_CUTOFF_DAY:
        txns_by_source_before_cutoff[t["source_account_id"]].append(t)

node_features = []
node_labels = []  # nhãn ở mức NODE (tài khoản), dùng để tham khảo, không dùng để train edge model
for acc in accounts_raw:
    aid = acc["account_id"]
    txns = txns_by_source_before_cutoff.get(aid, [])  # CHỈ lịch sử trước cutoff, nếu tài khoản đó chưa từng giao dịch trước cutoff (ví dụ mới mở tài khoản sau ngày 20), trả về list rỗng [] thay vì báo lỗi
    amounts = [t["amount"] for t in txns] #tách riêng list chỉ chứa số tiền của các giao dịch đó

    txn_count = len(txns) # số lượng giao dịch
    avg_amount = float(np.mean(amounts)) if amounts else 0.0 # trùng bình
    max_amount = float(np.max(amounts)) if amounts else 0.0 #   lớn nhất
    std_amount = float(np.std(amounts)) if len(amounts) > 1 else 0.0 # độ lệch chuẩn

    # ghép KYC vào để tạo 1 vector hoàn chỉnh
    kyc = acc["customer"]["kyc_detail"]
    feature_vector = [
        np.log1p(txn_count), # log(1+x), nén số lớn lại để chuẩn hóa min max để giảm đi ảnh hướng giữa só qá lớn và nhro
        np.log1p(avg_amount),
        np.log1p(max_amount),
        np.log1p(std_amount),
        kyc["face_match_score"], # lấy trức tiếp từ dữ liệu KYC ko cần cutoff
        kyc["attempt_no"],
        1.0 if kyc["liveness_status"] == "passed" else 0.0, #chuyển "passed"/"failed" (dạng chữ) thành số 1.0/0.0 (model không hiểu được chữ, chỉ hiểu số)
    ]
    node_features.append(feature_vector) #1 vector 7 con số - đây chính là xᵢ
    node_labels.append(1 if acc["is_fraud_account"] else 0) #chỉ để tham khảo/kiểm tra, không dùng để train

node_features = np.array(node_features, dtype=np.float32)

# Chuẩn hóa node feature (Min-Max scaling từng cột)
f_min = node_features.min(axis=0) #axis=0: tính min/max theo từng cột riêng biệt — ví dụ tìm min/max của cột "avg_amount" độc lập với cột "face_match_score"
f_max = node_features.max(axis=0)
node_features = (node_features - f_min) / (f_max - f_min + 1e-9)

x = torch.tensor(node_features, dtype=torch.float) #Chuyển sang tensor PyTorch

# 3. Build edge_index + edge_attr + nhãn (y) ở mức EDGE (giao dịch)
#    Bỏ giao dịch CASH_OUT không có dest_account_id (rút tiền mặt, không có node đích)
TYPE_LIST = ["TRANSFER", "PAYMENT", "CASH_OUT"] # danh sách cố định 3 loại giao dịch có trong dữ liệu, dùng để mã hóa one-hot

edge_src, edge_dst, edge_attr, edge_label, edge_day_list = [], [], [], [], []
# edge_src	Chỉ số (index) của tài khoản nguồn mỗi giao dịch
# edge_dst	Chỉ số của tài khoản đích
# edge_attr	Vector đặc trưng của giao dịch (amount, giờ, loại)
# edge_label	Nhãn 0/1 (có phải gian lận không)
# edge_day_list	Giao dịch này thuộc ngày thứ mấy

for t in all_txns: # duyệt qa tùng giao dịch để đóng gói
    src, dst = t["source_account_id"], t["dest_account_id"] 
    if dst is None or dst not in account_index:
        continue  # bỏ giao dịch cash-out ra ngoài hệ thống (không có node đích)

    src_i, dst_i = account_index[src], account_index[dst]  # chuyển chứ thành số

    #Trích xuất giờ và mã hóa one-hot cho loại giao dịch
    hour = datetime.fromisoformat(t["created_at"]).hour # lấy riêng phần giờ (0-23) từ created_at
    type_onehot = [1.0 if t["type"] == tp else 0.0 for tp in TYPE_LIST] #biến 1 giá trị dạng chữ ("TRANSFER") thành 1 vector số chỉ gồm 0 và 1.

    # Ghép mọi thứ lại, đẩy vào các list đã khai báo
    edge_src.append(src_i)
    edge_dst.append(dst_i)
    edge_attr.append([np.log1p(t["amount"]), hour / 23.0] + type_onehot)
    edge_label.append(1 if t["is_fraud"] else 0)
    edge_day_list.append(day_of(t))

#Chuyển tất cả từ list Python sang tensor PyTorch
edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long) # ghép 2 list edge_src, edge_dst thành 1 ma trận 2 hàng
edge_attr = torch.tensor(edge_attr, dtype=torch.float)  #dùng dtype=torch.float (số thực) vì đây là đặc trưng/nhãn, cần tính toán phép nhân, cộng, đạo hàm
y = torch.tensor(edge_label, dtype=torch.float)
edge_day = np.array(edge_day_list) #giữ nguyên NumPy, không chuyển sang tensor

# Chuẩn hóa edge_attr (2 cột đầu là số liên tục, giữ nguyên phần one-hot)
ea = edge_attr.numpy() #chuyển tạm tensor về NumPy để dễ thao tác cắt cột
ea_min = ea[:, :2].min(axis=0)
ea_max = ea[:, :2].max(axis=0)
ea[:, :2] = (ea[:, :2] - ea_min) / (ea_max - ea_min + 1e-9)
edge_attr = torch.tensor(ea, dtype=torch.float)
#KHÔNG chuẩn hóa 3 cột one-hot còn lại: vì one-hot vốn dĩ đã chỉ gồm 0 và 1 — chuẩn hóa Min-Max lên nó sẽ không đổi gì cả (0 và 1 đã nằm sẵn trong khoảng chuẩn), nên bỏ qua cho gọn code

print(f"Số cạnh (giao dịch dùng để train, đã bỏ cash-out ra ngoài hệ thống): {edge_index.shape[1]}")
print(f"Số cạnh gian lận: {int(y.sum().item())} ({y.mean().item()*100:.2f}%)")

data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y) # #"đóng gói" toàn bộ 4 thành phần, thành 1 object duy nhất tên data — object này chính là thứ được đưa thẳng vào model GATv2

# 4. Chia train/val/test THEO THỜI GIAN (không random) - đúng nguyên tắc dữ
#    liệu time-series, tránh dùng thông tin "tương lai" để đánh giá "quá khứ".
#    Ngày 0-19 (TRAIN_END_DAY): train | 20-24: validation | 25-29: test
n_edges = edge_index.shape[1]

train_mask = torch.tensor(edge_day < TRAIN_END_DAY, dtype=torch.bool)
val_mask = torch.tensor((edge_day >= TRAIN_END_DAY) & (edge_day < VAL_END_DAY), dtype=torch.bool)
test_mask = torch.tensor(edge_day >= VAL_END_DAY, dtype=torch.bool)

print(f"Mốc thời gian: Train = ngày 0-{TRAIN_END_DAY-1}, "
      f"Val = ngày {TRAIN_END_DAY}-{VAL_END_DAY-1}, Test = ngày {VAL_END_DAY} trở đi")
print(f"Train: {train_mask.sum().item()}, Val: {val_mask.sum().item()}, Test: {test_mask.sum().item()}")
print(f"Tỷ lệ fraud - Train: {y[train_mask].mean().item()*100:.2f}% | "
      f"Val: {y[val_mask].mean().item()*100:.2f}% | Test: {y[test_mask].mean().item()*100:.2f}%")



# 5. Định nghĩa model: 2 lớp GATv2Conv (encoder) + MLP decoder cho edge
class GATv2FraudModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, edge_dim, heads=4):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, edge_dim=edge_dim)
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=1, edge_dim=edge_dim)

        decoder_in = hidden_channels * 2 + edge_dim  # emb_source + emb_dest + edge_feature
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(decoder_in, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )

    def encode(self, x, edge_index, edge_attr):
        h = self.conv1(x, edge_index, edge_attr).relu()
        h = self.conv2(h, edge_index, edge_attr)
        return h  # embedding cuối cùng cho mỗi node

    def decode(self, h, edge_index, edge_attr):
        src, dst = edge_index
        emb_source = h[src]
        emb_dest = h[dst]
        z = torch.cat([emb_source, emb_dest, edge_attr], dim=-1)
        return self.decoder(z).squeeze(-1)  # logits (chưa qua sigmoid)

    def forward(self, x, edge_index, edge_attr):
        h = self.encode(x, edge_index, edge_attr)
        return self.decode(h, edge_index, edge_attr)


model = GATv2FraudModel(in_channels=x.shape[1], hidden_channels=16, edge_dim=edge_attr.shape[1], heads=4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

# Vì tỷ lệ fraud rất thấp (~1-2%), cần trọng số dương cao hơn để loss không "lười" đoán toàn 0
n_pos = y[train_mask].sum().item()
n_neg = train_mask.sum().item() - n_pos
pos_weight = torch.tensor([n_neg / max(n_pos, 1)])
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# 6. Vòng lặp train
def evaluate(mask, threshold=0.5):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_attr)
        probs = torch.sigmoid(logits)[mask].numpy()
        labels = data.y[mask].numpy()
        preds = (probs >= threshold).astype(int)

        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        try:
            auc = roc_auc_score(labels, probs)
        except ValueError:
            auc = float("nan")
    return precision, recall, f1, auc


N_EPOCHS = 100
for epoch in range(1, N_EPOCHS + 1):
    model.train()
    optimizer.zero_grad()
    logits = model(data.x, data.edge_index, data.edge_attr)
    loss = loss_fn(logits[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0 or epoch == 1:
        p, r, f1, auc = evaluate(val_mask)
        print(f"Epoch {epoch:3d} | Loss {loss.item():.4f} | "
              f"Val Precision {p:.3f} Recall {r:.3f} F1 {f1:.3f} AUC {auc:.3f}")

# 7. Đánh giá cuối cùng trên tập TEST
p, r, f1, auc = evaluate(test_mask)
print("\n=== KẾT QUẢ TRÊN TẬP TEST (GATv2), ngưỡng mặc định 0.5 ===")
print(f"Precision: {p:.3f}")
print(f"Recall:    {r:.3f}")
print(f"F1-score:  {f1:.3f}")
print(f"AUC-ROC:   {auc:.3f}")

# 8. Dò nhiều ngưỡng khác nhau (threshold sweep) để tìm điểm cân bằng
#    Precision/Recall tốt hơn ngưỡng mặc định 0.5
print("\n=== DÒ NGƯỠNG (threshold sweep) trên tập VALIDATION  ===")
print(f"{'Ngưỡng':>8} | {'Precision':>9} | {'Recall':>7} | {'F1':>6} | {'#Gắn cờ':>8}")
print("-" * 55)

model.eval()
with torch.no_grad():
    logits_all = model(data.x, data.edge_index, data.edge_attr)
    probs_val = torch.sigmoid(logits_all)[val_mask].numpy()
    labels_val = data.y[val_mask].numpy()

best_f1, best_threshold = -1.0, 0.5
thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.90, 0.95, 0.97, 0.99]

for th in thresholds:
    preds = (probs_val >= th).astype(int)
    p_th = precision_score(labels_val, preds, zero_division=0)
    r_th = recall_score(labels_val, preds, zero_division=0)
    f1_th = f1_score(labels_val, preds, zero_division=0)
    n_flagged = int(preds.sum())
    print(f"{th:>8.2f} | {p_th:>9.3f} | {r_th:>7.3f} | {f1_th:>6.3f} | {n_flagged:>8d}")

    if f1_th > best_f1:
        best_f1, best_threshold = f1_th, th

print(f"\n→ Ngưỡng cho F1 tốt nhất trong danh sách trên: {best_threshold} (F1 = {best_f1:.3f})")

# Có thể dò MỊN hơn quanh khoảng ngưỡng cao (nơi Precision thường cải thiện rõ nhất)
fine_thresholds = np.arange(0.90, 1.0, 0.01)
best_f1_fine, best_th_fine = -1.0, 0.5
for th in fine_thresholds:
    preds = (probs_val >= th).astype(int)
    f1_th = f1_score(labels_val, preds, zero_division=0)
    if f1_th > best_f1_fine:
        best_f1_fine, best_th_fine = f1_th, th

preds_best = (probs_val >= best_th_fine).astype(int)
p_best = precision_score(labels_val, preds_best, zero_division=0)
r_best = recall_score(labels_val, preds_best, zero_division=0)
print(f"\n=== NGƯỠNG TỐI ƯU (dò mịn 0.90-0.99) = {best_th_fine:.2f} ===")
print(f"Precision: {p_best:.3f} | Recall: {r_best:.3f} | F1: {best_f1_fine:.3f}")

# Áp dụng đúng 1 lần ngưỡng đã chọn (từ VAL) lên tập TEST — đây mới là con số
# báo cáo chính thức trong khóa luận, vì ngưỡng không hề "nhìn thấy" nhãn của TEST
probs_test_final = torch.sigmoid(logits_all)[test_mask].numpy()
labels_test_final = data.y[test_mask].numpy()
preds_test_final = (probs_test_final >= best_th_fine).astype(int)

print(f"\n=== KẾT QUẢ CUỐI CÙNG TRÊN TEST (ngưỡng {best_th_fine:.2f} chọn từ VAL) ===")
print(f"Precision: {precision_score(labels_test_final, preds_test_final):.3f}")
print(f"Recall:    {recall_score(labels_test_final, preds_test_final):.3f}")
print(f"F1:        {f1_score(labels_test_final, preds_test_final):.3f}")