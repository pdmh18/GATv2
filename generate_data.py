"""
Sinh dữ liệu giả lập (synthetic data) cho bài toán phát hiện giao dịch bất thường.
Theo đúng schema thiết kế: mỗi tài khoản = 1 document, có KYC + danh sách transactions.

Quy mô: 10,000 tài khoản, phân theo 4 nhóm hành vi (3 nhóm bình thường + 1 nhóm gian lận).
"""

import json
import random
from datetime import datetime, timedelta

import numpy as np
from mimesis import Person, Finance
from mimesis.locales import Locale

random.seed(42)
np.random.seed(42)

person = Person(Locale.EN)  # mimesis không có locale Tiếng Việt, dùng EN thay thế
finance = Finance()

SIM_START = datetime(2026, 3, 1)
SIM_DAYS = 30

# ---------------------------------------------------------------------------
# 1. Cấu hình các nhóm hành vi (persona)
# ---------------------------------------------------------------------------
N_ACCOUNTS = 10_000

PERSONAS = [
    {"name": "it_hoat_dong",   "ratio": 0.30, "poisson_lambda": 3,  "amount_mean": 13, "amount_sigma": 0.8},
    {"name": "binh_thuong",    "ratio": 0.50, "poisson_lambda": 15, "amount_mean": 14, "amount_sigma": 1.0},
    {"name": "hoat_dong_nhieu","ratio": 0.18, "poisson_lambda": 50, "amount_mean": 15, "amount_sigma": 1.2},
    {"name": "gian_lan",       "ratio": 0.02, "poisson_lambda": None, "amount_mean": None, "amount_sigma": None},
]

FRAUD_SCENARIOS = ["account_takeover", "mule_chain", "kyc_weak_abuse"]


# ---------------------------------------------------------------------------
# 2. Sinh giờ giao dịch theo phân phối 3 đỉnh (sáng / trưa / tối)
# ---------------------------------------------------------------------------
def gen_hour_normal():
    peak = np.random.choice(["morning", "lunch", "evening"], p=[0.35, 0.25, 0.40])
    if peak == "morning":
        h = np.random.normal(9, 1.5)
    elif peak == "lunch":
        h = np.random.normal(12, 1.0)
    else:
        h = np.random.normal(20, 1.5)
    return int(np.clip(h, 0, 23))


def gen_hour_fraud():
    # Giờ giấc bất thường: random đều cả ngày, không theo quy luật sinh hoạt bình thường
    return random.randint(0, 23)


def random_timestamp(day_offset, hour):
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return SIM_START + timedelta(days=day_offset, hours=hour, minutes=minute, seconds=second)


# ---------------------------------------------------------------------------
# 3. Sinh 1 tài khoản (accounts + KYC) theo persona
# ---------------------------------------------------------------------------
def gen_account(account_id, persona_name):
    is_fraud_acc = persona_name == "gian_lan"
    balance = round(np.random.lognormal(mean=15.5, sigma=1.0), -3)

    if is_fraud_acc:
        # KYC yếu hơn hẳn nhóm bình thường (dùng cho kịch bản kyc_weak_abuse,
        # nhưng vẫn gán chung cho mọi tài khoản gian lận để đơn giản hoá)
        face_match_score = round(random.uniform(0.5, 0.75), 2)
        attempt_no = random.randint(2, 4)
        liveness_status = random.choices(["passed", "failed"], weights=[0.7, 0.3])[0]
    else:
        face_match_score = round(random.uniform(0.90, 0.99), 2)
        attempt_no = 1
        liveness_status = "passed"

    opened_day_offset = random.randint(0, SIM_DAYS - 3)  # tài khoản mở trong khoảng mô phỏng

    account = {
        "account_id": account_id,
        "customer": {
            "full_name": person.full_name(),
            "date_of_birth": person.birthdate(min_year=1965, max_year=2005).isoformat(),
            "kyc_status": "verified",
            "kyc_detail": {
                "attempt_no": attempt_no,
                "face_match_score": face_match_score,
                "liveness_status": liveness_status,
            },
        },
        "balance": float(balance),
        "opened_at": (SIM_START + timedelta(days=opened_day_offset)).isoformat(),
        "persona": persona_name,
        "is_fraud_account": is_fraud_acc,
        "device_id": f"DEV_{random.randint(1, N_ACCOUNTS * 2):06d}",
        "transactions": [],  # sẽ điền ở bước sinh giao dịch
    }
    return account


# ---------------------------------------------------------------------------
# 4. Sinh giao dịch cho tài khoản BÌNH THƯỜNG
# ---------------------------------------------------------------------------
def gen_normal_transactions(account, all_account_ids, friend_map, txn_counter):
    # Tài khoản gian lận vẫn cần một nền giao dịch "bình thường" trước khi giở trò,
    # nên nếu persona là "gian_lan" (không có tham số riêng) thì mượn tạm persona "binh_thuong"
    persona_name = account["persona"] if account["persona"] != "gian_lan" else "binh_thuong"
    persona = next(p for p in PERSONAS if p["name"] == persona_name)
    n_txn = np.random.poisson(persona["poisson_lambda"])
    txns = []

    friends = friend_map[account["account_id"]]

    for _ in range(n_txn):
        # 70% giao dịch với "đối tác quen", 30% ngẫu nhiên
        if friends and random.random() < 0.7:
            dest = random.choice(friends)
        else:
            dest = random.choice(all_account_ids)

        amount = round(np.random.lognormal(mean=persona["amount_mean"], sigma=persona["amount_sigma"]), -3)
        amount = min(amount, account["balance"] * 0.5)  # không vượt quá 50% số dư (hành vi bình thường)
        amount = max(amount, 10_000)

        day_offset = random.randint(0, SIM_DAYS - 1)
        hour = gen_hour_normal()

        txn_counter[0] += 1
        txns.append({
            "transaction_id": f"TXN_{txn_counter[0]:07d}",
            "type": random.choice(["TRANSFER", "PAYMENT", "CASH_OUT"]),
            "amount": float(amount),
            "source_account_id": account["account_id"],
            "dest_account_id": dest,
            "created_at": random_timestamp(day_offset, hour).isoformat(),
            "device_id": account["device_id"],
            "is_fraud": False,
        })
    return txns


# ---------------------------------------------------------------------------
# 5. Sinh giao dịch cho tài khoản GIAN LẬN (3 kịch bản)
# ---------------------------------------------------------------------------
def gen_fraud_account_takeover(account, all_account_ids, txn_counter):
    """Hoạt động bình thường 1 thời gian, rồi bị chiếm: đổi thiết bị, transfer gần hết, cash_out ngay."""
    txns = gen_normal_transactions(account, all_account_ids, {account["account_id"]: []}, txn_counter)

    takeover_day = random.randint(20, SIM_DAYS - 2)
    hour = gen_hour_fraud()
    new_device = f"DEV_NEW_{random.randint(1, 99999):05d}"
    dest = random.choice(all_account_ids)

    transfer_amount = round(account["balance"] * random.uniform(0.8, 0.95), -3)
    txn_counter[0] += 1
    t1 = {
        "transaction_id": f"TXN_{txn_counter[0]:07d}",
        "type": "TRANSFER",
        "amount": float(transfer_amount),
        "source_account_id": account["account_id"],
        "dest_account_id": dest,
        "created_at": random_timestamp(takeover_day, hour).isoformat(),
        "device_id": new_device,
        "is_fraud": True,
    }

    txn_counter[0] += 1
    t2 = {
        "transaction_id": f"TXN_{txn_counter[0]:07d}",
        "type": "CASH_OUT",
        "amount": float(round(transfer_amount * 0.9, -3)),
        "source_account_id": dest,
        "dest_account_id": None,
        "created_at": random_timestamp(takeover_day, hour, ).isoformat(),
        "device_id": f"DEV_UNKNOWN_{random.randint(1,9999):04d}",
        "is_fraud": True,
    }
    txns.extend([t1, t2])
    return txns


def gen_fraud_kyc_weak_abuse(account, all_account_ids, txn_counter):
    """KYC yếu, giao dịch lớn bất thường ngay sau khi mở tài khoản/KYC xong."""
    opened_day = (datetime.fromisoformat(account["opened_at"]) - SIM_START).days
    hour = gen_hour_fraud()
    dest = random.choice(all_account_ids)

    amount = round(account["balance"] * random.uniform(0.7, 0.95), -3)
    txn_counter[0] += 1
    t1 = {
        "transaction_id": f"TXN_{txn_counter[0]:07d}",
        "type": "TRANSFER",
        "amount": float(amount),
        "source_account_id": account["account_id"],
        "dest_account_id": dest,
        "created_at": random_timestamp(opened_day, hour).isoformat(),
        "device_id": account["device_id"],
        "is_fraud": True,
    }
    # thêm vài giao dịch nhỏ bình thường xen kẽ để không quá lộ liễu
    normal_txns = gen_normal_transactions(account, all_account_ids, {account["account_id"]: []}, txn_counter)
    return normal_txns + [t1]


def gen_mule_chain(accounts_in_chain, all_account_ids, txn_counter):
    """Chuỗi 3-5 tài khoản chuyển tiền liên tiếp trong thời gian rất ngắn rồi rút hết."""
    start_day = random.randint(5, SIM_DAYS - 2)
    hour = gen_hour_fraud()
    base_time = random_timestamp(start_day, hour)

    txns_by_account = {acc["account_id"]: [] for acc in accounts_in_chain}
    current_amount = round(accounts_in_chain[0]["balance"] * random.uniform(0.85, 0.95), -3)

    for i in range(len(accounts_in_chain) - 1):
        src = accounts_in_chain[i]["account_id"]
        dst = accounts_in_chain[i + 1]["account_id"]
        step_time = base_time + timedelta(minutes=random.randint(2, 10) * (i + 1))

        txn_counter[0] += 1
        txns_by_account[src].append({
            "transaction_id": f"TXN_{txn_counter[0]:07d}",
            "type": "TRANSFER",
            "amount": float(current_amount),
            "source_account_id": src,
            "dest_account_id": dst,
            "created_at": step_time.isoformat(),
            "device_id": f"DEV_MULE_{random.randint(1,9999):04d}",
            "is_fraud": True,
        })
        current_amount = round(current_amount * random.uniform(0.85, 0.95), -3)

    # bước cuối: cash out
    last_acc = accounts_in_chain[-1]["account_id"]
    last_time = base_time + timedelta(minutes=random.randint(2, 10) * len(accounts_in_chain))
    txn_counter[0] += 1
    txns_by_account[last_acc].append({
        "transaction_id": f"TXN_{txn_counter[0]:07d}",
        "type": "CASH_OUT",
        "amount": float(current_amount),
        "source_account_id": last_acc,
        "dest_account_id": None,
        "created_at": last_time.isoformat(),
        "device_id": f"DEV_MULE_{random.randint(1,9999):04d}",
        "is_fraud": True,
    })
    return txns_by_account


# ---------------------------------------------------------------------------
# 6. Chạy toàn bộ pipeline sinh dữ liệu
# ---------------------------------------------------------------------------
def main():
    # 6.1 Gán persona cho từng tài khoản theo đúng tỷ lệ
    account_ids = [f"ACC_{i:06d}" for i in range(N_ACCOUNTS)]
    persona_pool = []
    for p in PERSONAS:
        persona_pool += [p["name"]] * int(N_ACCOUNTS * p["ratio"])
    while len(persona_pool) < N_ACCOUNTS:
        persona_pool.append("binh_thuong")
    random.shuffle(persona_pool)

    accounts = {aid: gen_account(aid, persona_pool[i]) for i, aid in enumerate(account_ids)}

    # 6.2 Gán "vòng tròn quen thuộc" (3-8 đối tác cố định) cho mỗi tài khoản
    friend_map = {
        aid: random.sample(account_ids, k=min(random.randint(3, 8), N_ACCOUNTS - 1))
        for aid in account_ids
    }

    txn_counter = [0]
    fraud_ids = [aid for aid, acc in accounts.items() if acc["is_fraud_account"]]
    random.shuffle(fraud_ids)

    # 6.3 Chia nhóm gian lận thành 3 kịch bản theo tỷ lệ đã thiết kế
    n_takeover = int(len(fraud_ids) * 0.40)
    n_kyc_abuse = int(len(fraud_ids) * 0.25)
    ids_takeover = fraud_ids[:n_takeover]
    ids_kyc_abuse = fraud_ids[n_takeover:n_takeover + n_kyc_abuse]
    ids_mule_pool = fraud_ids[n_takeover + n_kyc_abuse:]

    # Sinh giao dịch cho tài khoản BÌNH THƯỜNG
    for aid, acc in accounts.items():
        if not acc["is_fraud_account"]:
            acc["transactions"] = gen_normal_transactions(acc, account_ids, friend_map, txn_counter)

    # Sinh giao dịch cho kịch bản account_takeover
    for aid in ids_takeover:
        accounts[aid]["transactions"] = gen_fraud_account_takeover(accounts[aid], account_ids, txn_counter)
        accounts[aid]["fraud_scenario"] = "account_takeover"

    # Sinh giao dịch cho kịch bản kyc_weak_abuse
    for aid in ids_kyc_abuse:
        accounts[aid]["transactions"] = gen_fraud_kyc_weak_abuse(accounts[aid], account_ids, txn_counter)
        accounts[aid]["fraud_scenario"] = "kyc_weak_abuse"

    # Sinh giao dịch cho kịch bản mule_chain (nhóm 3-5 tài khoản 1 chuỗi)
    i = 0
    while i < len(ids_mule_pool):
        chain_len = min(random.randint(3, 5), len(ids_mule_pool) - i)
        chain_ids = ids_mule_pool[i:i + chain_len]
        if len(chain_ids) < 2:
            break
        chain_accounts = [accounts[cid] for cid in chain_ids]
        txns_by_account = gen_mule_chain(chain_accounts, account_ids, txn_counter)
        for cid in chain_ids:
            accounts[cid]["transactions"] = txns_by_account[cid]
            accounts[cid]["fraud_scenario"] = "mule_chain"
        i += chain_len

    # 6.4 Xuất ra file JSON
    output = list(accounts.values())
    with open("synthetic_banking_data.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    total_txns = sum(len(a["transactions"]) for a in output)
    print(f"Đã sinh {len(output)} tài khoản, {total_txns} giao dịch.")
    print(f"Số tài khoản gian lận: {len(fraud_ids)} "
          f"(takeover={len(ids_takeover)}, kyc_abuse={len(ids_kyc_abuse)}, mule_chain={len(ids_mule_pool)})")


if __name__ == "__main__":
    main()
