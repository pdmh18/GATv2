"""Report figure showing the actual feature encoding and tensor shapes."""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, default=ROOT/'outputs/contextual-v4/seed-20261002/features')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/contextual-v4/seed-20261002/figures')
    args = parser.parse_args()
    manifest = json.loads((args.features/'manifest.json').read_text())
    prep = json.loads((args.features/'preprocessing.json').read_text())
    feature_count = len(manifest['feature_names'])
    node_dim = len(manifest['node_feature_names'])
    column = manifest['feature_names'].index('log_amount')
    numeric = prep['numeric_indices'].index(column)
    log_amount = math.log1p(5000000)
    z = (log_amount-prep['mean'][numeric])/prep['scale'][numeric]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(15,10))
    ax.set(xlim=(0,15),ylim=(0,10));ax.axis('off')
    ax.text(7.5,9.65,'Biểu diễn dữ liệu giao dịch thành vector và graph',ha='center',fontsize=19,fontweight='bold',color='#0f172a')
    ax.text(7.5,9.18,'Mã hóa theo schema Phase 3 • Ví dụ minh họa, không phải một giao dịch thực',ha='center',fontsize=11,color='#475569')
    rows=[
        ['Trường dữ liệu','Ví dụ đầu vào','Biểu diễn số trong pipeline'],
        ['Amount','5.000.000 VND',f'ln(1 + amount) = {log_amount:.3f} → z = {z:.3f}'],
        ['Fee','0 VND','Giữ trong dữ liệu nguồn; bỏ khỏi input hiện hành vì không đổi'],
        ['Median cá nhân','TRANSFER: median 30 ngày = 200.000','Theo account + type, chỉ tính giao dịch đã hoàn tất trước t'],
        ['Amount tương đối','5 triệu / 200 nghìn = 25','ln(1 + 25) = 3,258; cần ≥5 lịch sử, thiếu thì 0 + cờ'],
        ['Type','PAYMENT','[0, 0, 0, 1, 0, 0] — one-hot 6 chiều'],
        ['Channel','web','[0, 1, 0, 0] — one-hot 4 chiều'],
        ['Giờ giao dịch','06:00','[sin(2π × 6/24), cos(2π × 6/24)] ≈ [1, 0]'],
        ['Tần suất gần đây','6 requests / 120 giây','ln(1 + 6) = 1,946 → chuẩn hóa bằng train mean/std'],
        ['Người nhận mới','Chưa từng gặp trong lịch sử','1; nếu đã gặp thì 0'],
        ['Loại node','account','[0, 1, 0, 0] — one-hot 4 chiều'],
        ['Account ID','ACC_A, ACC_B, ACC_C','Chỉ số node: 12, 25, 38 — dùng để nối graph'],
        ['Nhãn đánh giá','normal / anomaly','0 / 1 — target riêng, không ghép vào feature vector'],
    ]
    table=ax.table(cellText=rows,cellLoc='left',colWidths=[.20,.26,.54],bbox=[.025,.365,.95,.50])
    table.auto_set_font_size(False);table.set_fontsize(11)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor('#cbd5e1');cell.set_linewidth(.7);cell.PAD=.06
        cell.set_facecolor('#dbeafe' if r==0 else ('#f8fafc' if r%2 else 'white'))
        if r==0:cell.set_text_props(weight='bold',color='#172554')
    ax.text(.45,3.34,'Thứ tự type: [TRANSFER, CASH_IN, CASH_OUT, PAYMENT, DEBIT, unknown]',fontsize=10,color='#475569')
    ax.text(.45,3.06,'Thứ tự channel: [mobile, web, desktop, unknown] • Loại node: [unknown, account, merchant, cash]',fontsize=10,color='#475569')
    def panel(x,y,w,h,title,body,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.12',facecolor=color,edgecolor='#94a3b8'))
        ax.text(x+w/2,y+h-.20,title,ha='center',va='top',fontsize=12,fontweight='bold',color='#172554')
        ax.text(x+w/2,y+.47,body,ha='center',va='center',fontsize=11,linespacing=1.5,color='#1e293b')
    panel(.45,1.15,6.65,1.52,'Feature giao dịch',
          f'Một giao dịch → vector {feature_count} số\nN giao dịch → ma trận [N, {feature_count}]\nType/channel không dùng số thứ tự 1, 2, 3…','#eff6ff')
    panel(7.9,1.15,6.65,1.52,'Graph và node features',
          f'A → B, A → C: edge_index = [[12, 12], [25, 38]]\nHàng 1: nguồn • Hàng 2: đích • Shape [2, E]\nNode features: [số node, {node_dim}] = 4 loại + 8 lịch sử','#ecfdf5')
    ax.text(7.5,.65,'z = (giá trị log − mean_train) / std_train • Ví dụ z của amount dùng preprocessing artifact hiện hành.',
            ha='center',fontsize=10,color='#475569')
    ax.text(7.5,.30,'Lịch sử chỉ dùng thông tin trước thời điểm chấm. CASH_IN đảo chiều trong graph dòng tiền; target giữ vai trò source/destination gốc.',
            ha='center',fontsize=9.5,color='#475569')
    fig.tight_layout(pad=.6);args.output.mkdir(parents=True,exist_ok=True)
    for ext in ('png','svg','pdf'):
        fig.savefig(args.output/f'feature_encoding.{ext}',dpi=240,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    print(args.output/'feature_encoding.png')


if __name__=='__main__':
    main()
