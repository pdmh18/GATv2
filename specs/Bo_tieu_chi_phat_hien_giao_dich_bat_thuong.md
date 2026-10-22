# Bộ tiêu chí phát hiện giao dịch bất thường — phiên bản 2

## 1. Mục đích và phạm vi

Tài liệu là đặc tả nghiên cứu cho transaction monitoring và mô hình GATv2. Một tiêu chí khớp là một quan sát cần đánh giá trong ngữ cảnh; không tự trở thành nhãn anomaly, bằng chứng gian lận hoặc quyết định chặn giao dịch. Các ngưỡng là giả thuyết thử nghiệm, chưa được hiệu chỉnh từ dữ liệu ngân hàng thật.

Giữ nguyên 49 mã A01–H05 để truy vết bản đầu. Catalog máy đọc được tại [anomaly_criteria.v2.json](anomaly_criteria.v2.json); đây là nguồn định nghĩa tiêu chí, tham số, yêu cầu dữ liệu và mức hỗ trợ hiện tại. Bảng mục 4 được sinh từ catalog bằng `scripts/render_criteria_spec.py`.

Phân biệt ba lớp: **behavior_signal** (dấu hiệu quan sát), **external_fact** (kết quả tra cứu/xác minh có nguồn), **contextual_review** (cần hồ sơ hoặc đánh giá nghiệp vụ). GATv2 có thể học từ feature quan sát được; dữ kiện đã xác minh dùng đúng giá trị as-of, không bắt model đoán lại.

## 2. Hợp đồng thời gian, dữ liệu và nhãn

### 2.1. Đơn vị và thời điểm

Mỗi kết quả phải có criterion_id/version, entity_id và unit, evaluated_at, evidence_event_ids, event_time cutoff, data_watermark, giá trị đo, threshold, support và trạng thái. Unit phân biệt transaction, account_window, device_window, entity_group và customer_or_account. Các tiêu chí end_of_day chỉ đánh giá sau khi ngày kết thúc, không áp ngược kết quả cho request buổi sáng.

- `matched`: đủ dữ liệu, mọi điều kiện bắt buộc của nhánh đều đúng.
- `not_matched`: đủ dữ liệu nhưng điều kiện không thỏa.
- `insufficient_data`: thiếu source, watermark, support hoặc tham số bắt buộc. Không thay bằng normal/false.

Request-time tại t chỉ thấy event_time < t và dữ liệu đã available_at <= t. Các request cùng timestamp dùng chung prior state. Watchlist, hồ sơ, kết quả KYC phải thỏa cả thời điểm hiệu lực và thời điểm hệ thống biết; bản ghi được thêm sau t không được dùng hồi tố cho feature tại t. Event-time replay giả định lịch sử đã đầy đủ; production cần watermark/late-event policy.

### 2.2. Ngữ nghĩa giao dịch và cửa sổ

- Request count đếm yêu cầu, bao gồm yêu cầu sau này thất bại. Loại trùng theo event_id; xung đột cùng ID phải báo lỗi. Mỗi request tương ứng một transaction_id, không đếm terminal thành request mới.
- Settled flow chỉ tính TransferCompleted/TransactionPosted đã xảy ra trước cutoff. Failed/cancelled không làm tăng tiền đã nhận/chuyển. Số nguồn/đích khác nhau phải đếm distinct account, không đếm số giao dịch.
- Với dòng tiền, TRANSFER theo source→destination; CASH_IN theo cash→account; CASH_OUT/PAYMENT/DEBIT theo account→counterparty. Các công thức C chỉ định rõ phạm vi, không ngầm coi mọi loại nghiệp vụ là chuyển khoản.
- Amount không gồm fee trừ khi ghi rõ. Số dư phải đối chiếu mọi bút toán, phí, hoàn tiền và funding; snapshot transaction riêng không bảo đảm toàn bộ sổ cái đã đầy đủ.
- Rolling window là [t-W,t). `day` là ngày lịch Asia/Ho_Chi_Minh khi ghi end_of_day; 30d là 30×24h, không phải tháng lịch.
- Request branch được phép cộng count/amount hiện tại vào **projected** metric. Không gọi projected amount là settled amount và không cập nhật lịch sử thành công trước kết quả.
- Baseline tổng theo kỳ được so với các kỳ cùng độ dài, nằm hoàn toàn trước cửa sổ hiện tại. Bao gồm kỳ không có giao dịch khi đã quan sát đầy đủ. Không lấy median amount từng giao dịch thay cho median tổng tiền ngày/tuần.
- Sai lệch amount tương đối được tính bằng |a-b|/max(a,b) với a,b>0. Mẫu số bằng 0, lịch sử chưa đủ kỳ/support hoặc giá trị tối thiểu chưa cấu hình (`null`) → insufficient_data cho nhánh tương ứng. Không tự đặt epsilon hoặc threshold từ test để khiến tiêu chí khớp.
- `new_receiver`/`new_device` nghĩa là chưa xuất hiện trong lịch sử đã quan sát. Thiếu lịch sử tài khoản hoặc danh mục thiết bị ban đầu không chứng minh đối tác/thiết bị mới thực sự.

Episode C03/C05 dùng cửa sổ cố định để tái lập: mở tại settled incoming TRANSFER đầu tiên chưa thuộc episode nào, thu nhận incoming trong 30 phút, sau đó quan sát outgoing trong tối đa 60 phút. Đóng khi lần đầu đủ điều kiện C03 hoặc hết hạn; episode tiếp theo mở bằng incoming mới sau khi episode trước đóng, không dùng lại transaction. Phạm vi này chưa bao phủ mọi kiểu nhận/chuyển đan xen; cần kịch bản riêng nếu mở rộng. Watchlist có `effective_to=null` được hiểu là chưa có thời điểm hết hiệu lực, nhưng vẫn phải kiểm tra known_at.

### 2.3. Nhãn và kịch bản

Lưu riêng criterion_result, scenario_id/entity_roles, anomaly_label và verified_fraud_label. Label gian lận có thể được xác minh sau t; điều cấm là dùng thông tin sau t làm feature. Khi mục tiêu là cảnh báo theo chuỗi, phải ghi onset_at/alertable_at và xác định giao dịch/vai trò nào đang được chấm, không gắn toàn bộ tài khoản tham gia thành fraud mặc định.

Generator tạo nguyên nhân và chuỗi sự kiện trước; nhãn xuất phát từ kịch bản và vai trò được thiết kế. Không dùng `matched == anomaly_label`. Mỗi kịch bản cần trường hợp bình thường dễ nhầm: merchant/thu hộ, payroll, chi trả định kỳ, đổi điện thoại, thiết bị dùng chung hợp lệ. Không đưa scenario, profile nội bộ generator hoặc nhãn vào tensor đầu vào.

## 3. Tổng hợp cảnh báo và vai trò mô hình

Bỏ bảng cộng điểm minh họa 135 điểm của phiên bản đầu. Chưa có trọng số, cutoff 30/50/70 hoặc phép quy đổi điểm luật được xác nhận. Giữ `risk_score` hiện có là output model; `criterion_results` và backend `decision` là các thông tin riêng, không cộng điểm luật vào sigmoid. Quyết định giữ/xác minh giao dịch cần policy version, lý do, thời hạn và xử lý timeout tại backend trước khi hoàn tất nếu nghiệp vụ yêu cầu.

Các tiêu chí có bằng chứng trùng nhau phải liên kết episode/evidence IDs để tránh cộng trùng. Feature mới cần kiểm tra tương quan, sensitivity/ablation trên GATv2 và cảnh báo sai theo nhóm; không xóa hoặc thêm feature chỉ để tăng điểm trên test đã dùng.

## 4. Catalog tiêu chí

Các tham số số cụ thể là cấu hình đề xuất, không phải ngưỡng được chứng minh tối ưu. Cột hỗ trợ model không đồng nghĩa tiêu chí đã được triển khai đầy đủ. `minimum_calendar_history_days` chỉ là điều kiện lịch cần thiết; EDA không coi đủ số ngày là đủ support hoặc đủ watermark.

<!-- CRITERIA_CATALOG_START -->

### A. Giá trị

| Mã | Tiêu chí / đơn vị | Thời điểm / loại | Công thức |
| --- | --- | --- | --- |
| A01 | Giá trị một giao dịch tăng đột biến; transaction | Request / giám sát; behavior_signal | amount >= minimum_amount AND amount / median(completed amounts cùng source+type trong [t-90d,t)) > ratio; cần đủ support. |
| A02 | Tổng giá trị trong 24 giờ bất thường; account_window | Request / giám sát; behavior_signal | Tổng settled tiền vào và tiền ra được xét riêng trên [t-24h,t); so với median 28 cửa sổ 24h liền trước, không chồng cửa sổ hiện tại. Điều kiện định lượng: total >= minimum_amount AND total/reference_median > ratio, reference_median>0. |
| A03 | Tổng giá trị trong 7 ngày tăng bất thường; account_window | Request / giám sát; behavior_signal | Tổng settled vào/ra riêng trong [t-7d,t); so median 12 kỳ 7d liền trước [t-91d,t-7d). Điều kiện định lượng: total >= minimum_amount AND total/reference_median > ratio, reference_median>0. |
| A04 | Tổng giao dịch trong 30 ngày tăng đột biến; account_window | Request / giám sát; behavior_signal | Tổng settled vào/ra riêng trên 30d; so median 6 kỳ 30d trước đó. Cần hồ sơ chu kỳ kinh doanh/thu nhập để xác nhận ngữ cảnh. Điều kiện định lượng: total >= minimum_amount AND total/reference_median > ratio, reference_median>0. |

- **A01** — nguồn: `request_history, settled_history`; lịch tối thiểu: 90 ngày; tham số: `{"minimum_amount": 10000000, "ratio": 5, "reference_days": 90, "min_reference_transactions": 5}`. Trường hợp hợp lệ dễ nhầm: Khách mua tài sản hoặc thanh toán khoản lớn đã có ngữ cảnh hợp lệ. Hỗ trợ hiện tại: Đã có amount/median theo source+type nhưng reference hiện tại 30d, không phải A01-90d.
- **A02** — nguồn: `settled_history`; lịch tối thiểu: 29 ngày; tham số: `{"window_seconds": 86400, "reference_periods": 28, "ratio": 3, "minimum_amount": null}`. Trường hợp hợp lệ dễ nhầm: Ngày trả lương, thu hộ hoặc doanh thu tăng đã được giải thích. Hỗ trợ hiện tại: Cần bổ sung feature/công thức; không phải luật đã triển khai.
- **A03** — nguồn: `settled_history`; lịch tối thiểu: 91 ngày; tham số: `{"window_seconds": 604800, "reference_periods": 12, "ratio": 3, "minimum_amount": null}`. Trường hợp hợp lệ dễ nhầm: Chi trả định kỳ hoặc doanh thu theo mùa. Hỗ trợ hiện tại: Cần bổ sung feature/công thức; không phải luật đã triển khai.
- **A04** — nguồn: `settled_history, observed_profile`; lịch tối thiểu: 210 ngày; tham số: `{"window_days": 30, "reference_periods": 6, "ratio": 3, "minimum_amount": null}`. Trường hợp hợp lệ dễ nhầm: Doanh thu theo mùa, khoản thanh toán định kỳ lớn. Hỗ trợ hiện tại: Cần bổ sung feature/công thức; không phải luật đã triển khai.

### B. Tần suất

| Mã | Tiêu chí / đơn vị | Thời điểm / loại | Công thức |
| --- | --- | --- | --- |
| B01 | Số lượng giao dịch trong thời gian ngắn tăng cao; account_window | Request / giám sát; behavior_signal | count prior requests 1h + request hiện tại >= count AND vượt ratio lần median count của 28 khoảng cùng giờ địa phương trong 28 ngày trước. |
| B02 | Tần suất giao dịch trong ngày bất thường; account_window | Request / giám sát; behavior_signal | count prior requests 24h + request hiện tại >= count; median count 28 kỳ 24h liền trước < baseline_max. |
| B03 | Nhận tiền từ nhiều nguồn trong thời gian ngắn; account_window | Sau settled / giám sát; behavior_signal | distinct settled TRANSFER source tới account trong [t-30m,t) >= sources AND settled incoming amount >= minimum_amount. |
| B04 | Chuyển tiền liên tiếp cho nhiều người nhận; account_window | Request / giám sát; behavior_signal | distinct recipients của settled outgoing TRANSFER trên 1h (hoặc projected khi request TRANSFER hiện tại thành công) >= recipients và vượt baseline 28 kỳ. |
| B05 | Giao dịch dồn dập; account_window | Request / giám sát; behavior_signal | count prior requests 5m + request hiện tại >= count AND tổng amount yêu cầu 5m gồm hiện tại >= minimum_amount. Attempt không đồng nghĩa tiền đã chuyển. |
| B06 | Tái hoạt động sau thời gian dài không giao dịch; account_window | Sau settled / giám sát; behavior_signal | Đã quan sát tài khoản >=90d; trước đợt mới không có request >=90d; trong 1h đầu tái hoạt động có >=5 settled sources và >=3 settled outgoing transfers. |

- **B01** — nguồn: `request_history`; lịch tối thiểu: 29 ngày; tham số: `{"window_seconds": 3600, "count": 10, "ratio": 3, "reference_days": 28}`. Trường hợp hợp lệ dễ nhầm: Merchant xử lý đơn trong giờ cao điểm. Hỗ trợ hiện tại: Có count request 1h; chưa có baseline cùng giờ 28 ngày.
- **B02** — nguồn: `request_history`; lịch tối thiểu: 29 ngày; tham số: `{"window_seconds": 86400, "count": 30, "baseline_max": 5, "reference_periods": 28}`. Trường hợp hợp lệ dễ nhầm: Ngày mở bán hoặc kỳ thu phí. Hỗ trợ hiện tại: Cần bổ sung feature/công thức; không phải luật đã triển khai.
- **B03** — nguồn: `settled_history, directed_timeline`; lịch tối thiểu: 0 ngày; tham số: `{"window_seconds": 1800, "sources": 5, "minimum_amount": null}`. Trường hợp hợp lệ dễ nhầm: Cửa hàng hoặc thu hộ nhận nhiều người trả. Hỗ trợ hiện tại: Có distinct transfer attempts 10m, không tương đương settled inflow 30m.
- **B04** — nguồn: `request_history, settled_history, directed_timeline`; lịch tối thiểu: 29 ngày; tham số: `{"window_seconds": 3600, "recipients": 10, "ratio": 3, "reference_periods": 28}`. Trường hợp hợp lệ dễ nhầm: Trả lương hoặc chi trả nhiều đối tác. Hỗ trợ hiện tại: Cần bổ sung feature/công thức; không phải luật đã triển khai.
- **B05** — nguồn: `request_history`; lịch tối thiểu: 0 ngày; tham số: `{"window_seconds": 300, "count": 5, "minimum_amount": null}`. Trường hợp hợp lệ dễ nhầm: Tách thanh toán hợp lệ hoặc thao tác lại sau lỗi. Hỗ trợ hiện tại: Có prior request count 300s; phải cộng current khi đánh giá request.
- **B06** — nguồn: `request_history, settled_history, account_lifecycle`; lịch tối thiểu: 90 ngày; tham số: `{"dormant_days": 90, "episode_seconds": 3600, "sources": 5, "outgoing_count": 3}`. Trường hợp hợp lệ dễ nhầm: Tài khoản ngủ đông được dùng lại cho mục đích hợp lệ. Hỗ trợ hiện tại: Cần bổ sung feature/công thức; không phải luật đã triển khai.

### C. Luân chuyển tiền

| Mã | Tiêu chí / đơn vị | Thời điểm / loại | Công thức |
| --- | --- | --- | --- |
| C01 | Chuyển đi nhanh sau khi nhận tiền; account_window | Request / giám sát; behavior_signal | R60 = settled outgoing TRANSFER+CASH_OUT amount / settled incoming TRANSFER amount trong [t-60m,t); R60>=ratio và inflow>=minimum_inflow. Đây là tỷ lệ luồng tiền, không khẳng định truy nguyên tiền. Request variant thêm amount hiện tại vào tử projected, không sửa settled numerator. |
| C02 | Chuyển tiếp phần lớn tiền trong ngày; account_window | Cuối ngày; behavior_signal | Sau khi ngày địa phương D kết thúc: outflow TRANSFER+CASH_OUT / inflow TRANSFER trong D >=ratio, inflow>=minimum_inflow và closing_balance/inflow < remaining_ratio. |
| C03 | Nhiều nguồn tiền vào, ít nguồn tiền ra; account_window | Sau settled / giám sát; behavior_signal | Trong 30m trước đợt nhận kết thúc có >=sources settled source khác nhau; sau đó trong <=60m có 1..max_recipients settled outgoing recipients và out/in >=ratio. |
| C04 | Tài khoản trung chuyển có số dư thấp; account_window | Cuối ngày; behavior_signal | Cuối ngày: closing_balance / settled inflow TRANSFER <ratio, inflow>=minimum_inflow, distinct sources>=sources; ít nhất repeat_days ngày trong 7d. |
| C05 | Chuỗi nhận và chuyển tiếp lặp lại; account_window | Sau settled / giám sát; behavior_signal | Số episode C03 matched đã đóng, không trùng transaction_id, trong 7d >= episodes. Episode mới chỉ mở sau episode trước kết thúc. |
| C06 | Rút tiền mặt ngay sau khi nhận; account_window | Request / giám sát; behavior_signal | Settled CASH_OUT amount trong 60m / settled incoming TRANSFER amount trong cùng 60m >=ratio và inflow>=minimum_inflow. Đánh giá projected riêng tại request CASH_OUT. |

- **C01** — nguồn: `settled_history, directed_timeline`; lịch tối thiểu: 0 ngày; tham số: `{"window_seconds": 3600, "ratio": 0.8, "minimum_inflow": null}`. Trường hợp hợp lệ dễ nhầm: Thu hộ rồi chi hộ, chuyển tiền giữa tài khoản của chính khách. Hỗ trợ hiện tại: Có received settled TRANSFER 10m và outgoing attempts; chưa có R60 settled/projected.
- **C02** — nguồn: `settled_history, balance_snapshots`; lịch tối thiểu: 0 ngày; tham số: `{"ratio": 0.9, "remaining_ratio": 0.1, "minimum_inflow": null}`. Trường hợp hợp lệ dễ nhầm: Thu hộ hoặc thanh toán nghĩa vụ trong ngày. Hỗ trợ hiện tại: Cần bổ sung feature/công thức; không phải luật đã triển khai.
- **C03** — nguồn: `settled_history, directed_timeline`; lịch tối thiểu: 0 ngày; tham số: `{"incoming_seconds": 1800, "forward_seconds": 3600, "sources": 5, "max_recipients": 3, "ratio": 0.8}`. Trường hợp hợp lệ dễ nhầm: Đơn vị thu hộ tập trung và thanh toán cho nhà cung cấp. Hỗ trợ hiện tại: Cần bổ sung feature/công thức; không phải luật đã triển khai.
- **C04** — nguồn: `settled_history, balance_snapshots`; lịch tối thiểu: 7 ngày; tham số: `{"ratio": 0.1, "sources": 5, "repeat_days": 3, "minimum_inflow": null}`. Trường hợp hợp lệ dễ nhầm: Tài khoản chuyên thu/chi có số dư mục tiêu thấp. Hỗ trợ hiện tại: Cần bổ sung feature/công thức; không phải luật đã triển khai.
- **C05** — nguồn: `settled_history, directed_timeline`; lịch tối thiểu: 7 ngày; tham số: `{"window_days": 7, "episodes": 3}`. Trường hợp hợp lệ dễ nhầm: Quy trình thu hộ/chi hộ lặp định kỳ. Hỗ trợ hiện tại: Cần bổ sung feature/công thức; không phải luật đã triển khai.
- **C06** — nguồn: `settled_history, directed_timeline`; lịch tối thiểu: 0 ngày; tham số: `{"window_seconds": 3600, "ratio": 0.8, "minimum_inflow": null}`. Trường hợp hợp lệ dễ nhầm: Rút tiền chi tiêu sau khi nhận lương. Hỗ trợ hiện tại: Cần bổ sung feature/công thức; không phải luật đã triển khai.

### D. Đối tác / mạng lưới

| Mã | Tiêu chí / đơn vị | Thời điểm / loại | Công thức |
| --- | --- | --- | --- |
| D01 | Nhận tiền từ đối tác có trong danh sách cảnh báo; transaction | Request / giám sát; external_fact | Counterparty của giao dịch có watchlist record đã known_at<=t và effective_from<=t<effective_to, trạng thái active. |
| D02 | Nhận tiền từ nhiều tài khoản đáng ngờ; account_window | Sau settled / giám sát; external_fact | Có >=sources distinct settled incoming sources với watchlist active as-of từng thời điểm nhận trong 24h; không đếm nhiều giao dịch cùng nguồn thành nhiều nguồn. |
| D03 | Số lượng đối tác mới tăng đột biến; account_window | Request / giám sát; behavior_signal | distinct counterparties lần đầu quan sát trong 24h >=count và >ratio lần baseline median 28 kỳ 24h trước. Known-before-window lấy toàn prior history, cold start là unknown. |
| D04 | Giao dịch tập trung vào một đối tác bất thường; account_window | Request / giám sát; behavior_signal | max recipient settled outgoing TRANSFER share trong 24h >share; recipient mới trước cửa sổ hoặc watchlisted as-of t. Nhánh mới và watchlist báo riêng, không dùng OR để che thiếu watchlist. |
| D05 | Quan hệ chuyển tiền khép kín; account_window | Sau settled / giám sát; behavior_signal | Có >=cycles chu trình 3 node A→B→C→A, cạnh settled TRANSFER với timestamp tăng nghiêm ngặt và span<=7d. Chu trình đếm không trùng transaction; mục đích kinh tế chưa thể suy ra. |
| D06 | Giao dịch đảo chiều thường xuyên; account_window | Sau settled / giám sát; behavior_signal | Trong 24h, một cặp có >=round_trips cặp settled A→B rồi B→A; ghép FIFO không dùng lại tx, mỗi cặp cách nhau<=1h và relative amount difference<=tolerance. |
| D07 | Một tài khoản nhận rồi phân phối lại; account_window | Sau settled / giám sát; behavior_signal | Trong 24h có >=sources distinct settled TRANSFER sources và >=recipients distinct settled outgoing recipients; báo cùng out/in ratio và độ trễ, không suy ra mục đích. |
| D08 | Mạng lưới tài khoản có rủi ro liên quan; account_window | Request / giám sát; external_fact | Có >=neighbors đối tác 1-hop trên settled directed graph 7d mang cảnh báo đã biết tại t; chỉ lan truyền từ evidence as-of, không lan từ anomaly labels offline. |

- **D01** — nguồn: `watchlist_asof`; lịch tối thiểu: 0 ngày; tham số: `{}`. Trường hợp hợp lệ dễ nhầm: Danh sách cũ, nhầm định danh hoặc kết quả đã được gỡ. Hỗ trợ hiện tại: Cần bổ sung feature/công thức; không phải luật đã triển khai.
- **D02** — nguồn: `watchlist_asof, settled_history, directed_timeline`; lịch tối thiểu: 0 ngày; tham số: `{"window_seconds": 86400, "sources": 4}`. Trường hợp hợp lệ dễ nhầm: Một đối tác có cảnh báo cần rà soát, chưa đủ kết luận bên nhận. Hỗ trợ hiện tại: Cần bổ sung feature/công thức; không phải luật đã triển khai.
- **D03** — nguồn: `request_history, directed_timeline`; lịch tối thiểu: 29 ngày; tham số: `{"window_seconds": 86400, "count": 10, "ratio": 3, "reference_periods": 28}`. Trường hợp hợp lệ dễ nhầm: Khách mới mở cửa hàng hoặc tăng tập đối tác hợp lệ. Hỗ trợ hiện tại: Có new_receiver từng request; chưa có distinct new counterparties 24h và baseline.
- **D04** — nguồn: `settled_history, directed_timeline`; lịch tối thiểu: 0 ngày; tham số: `{"window_seconds": 86400, "share": 0.8}`. Trường hợp hợp lệ dễ nhầm: Thanh toán nhà cung cấp mới hoặc mua tài sản. Hỗ trợ hiện tại: Cần bổ sung feature/công thức; không phải luật đã triển khai.
- **D05** — nguồn: `settled_history, directed_timeline`; lịch tối thiểu: 0 ngày; tham số: `{"max_span_days": 7, "cycles": 2}`. Trường hợp hợp lệ dễ nhầm: Hoàn trả, phân bổ quỹ hoặc luân chuyển nội bộ hợp lệ. Hỗ trợ hiện tại: Graph model bidirectional/deduplicated không giữ đủ chiều/thứ tự để xác nhận D05.
- **D06** — nguồn: `settled_history, directed_timeline`; lịch tối thiểu: 0 ngày; tham số: `{"window_seconds": 86400, "pair_seconds": 3600, "round_trips": 3, "tolerance": 0.1}`. Trường hợp hợp lệ dễ nhầm: Hoàn tiền hoặc điều chuyển hợp lệ giữa hai tài khoản. Hỗ trợ hiện tại: Cần bổ sung feature/công thức; không phải luật đã triển khai.
- **D07** — nguồn: `settled_history, directed_timeline`; lịch tối thiểu: 0 ngày; tham số: `{"window_seconds": 86400, "sources": 10, "recipients": 5}`. Trường hợp hợp lệ dễ nhầm: Payroll, sàn thanh toán hoặc tài khoản thu hộ. Hỗ trợ hiện tại: Cần bổ sung feature/công thức; không phải luật đã triển khai.
- **D08** — nguồn: `watchlist_asof, settled_history, directed_timeline`; lịch tối thiểu: 0 ngày; tham số: `{"window_days": 7, "neighbors": 3}`. Trường hợp hợp lệ dễ nhầm: Quan hệ gián tiếp không chứng minh chủ tài khoản có hành vi sai. Hỗ trợ hiện tại: Cần bổ sung feature/công thức; không phải luật đã triển khai.

### E. Số dư / dòng tiền

| Mã | Tiêu chí / đơn vị | Thời điểm / loại | Công thức |
| --- | --- | --- | --- |
| E01 | Vòng quay tiền tăng bất thường; account_window | Cuối ngày; behavior_signal | Sau ngày kết thúc: gross settled inflow+outflow / median gross 28 ngày trước >=ratio, opening/closing balance biến động<=balance_delta_ratio; cần loại trừ điều chỉnh sổ cái. |
| E02 | Số dư không tương xứng với quy mô giao dịch; account_window | Cuối ngày; behavior_signal | Gross settled flow 24h / median opening balance 28 ngày trước >=ratio, baseline balance>0; lặp >=repeat_days trong 7d và xét hồ sơ hoạt động. |
| E03 | Nhận tiền nhưng không có hành vi chi tiêu thông thường; account_window | Rà soát định kỳ; contextual_review | Trong 30d có >=sources distinct settled incoming sources và PAYMENT+DEBIT spend / settled total outflow <spend_share; cần mục đích tài khoản và khả năng quan sát chi tiêu ngoài hệ thống. |
| E04 | Nguồn tiền không tương xứng hồ sơ; account_window | Rà soát định kỳ; contextual_review | Settled inflow 30d / mức thu nhập hoặc doanh thu đã xác minh, có hiệu lực tại t, >ratio; thiếu hồ sơ là insufficient_data. |
| E05 | Tiền vào và ra tương ứng bất thường; account_window | Sau settled / giám sát; behavior_signal | Ghép FIFO không dùng lại settled incoming TRANSFER và outgoing TRANSFER/CASH_OUT cùng account, gap<=1h, amount difference<=tolerance; >=pairs cặp trong 24h. Không gọi là truy nguyên tiền. |
| E06 | Hoạt động dồn vào một số khoảng ngắn; account_window | Request / giám sát; behavior_signal | max request count của cửa sổ 5m trong 24h / count requests 24h >=share và tổng count>=min_count; so với baseline 28 ngày trước. |

- **E01** — nguồn: `settled_history, balance_snapshots`; lịch tối thiểu: 29 ngày; tham số: `{"reference_days": 28, "ratio": 3, "balance_delta_ratio": 0.1}`. Trường hợp hợp lệ dễ nhầm: Tài khoản thanh toán trung gian có vòng quay cao hợp lệ. Hỗ trợ hiện tại: Cần bổ sung feature/công thức; không phải luật đã triển khai.
- **E02** — nguồn: `settled_history, balance_snapshots, observed_profile`; lịch tối thiểu: 35 ngày; tham số: `{"reference_days": 28, "ratio": 10, "repeat_days": 3}`. Trường hợp hợp lệ dễ nhầm: Merchant có số dư mục tiêu thấp. Hỗ trợ hiện tại: Cần bổ sung feature/công thức; không phải luật đã triển khai.
- **E03** — nguồn: `settled_history, observed_profile, purpose_evidence`; lịch tối thiểu: 30 ngày; tham số: `{"window_days": 30, "sources": 5, "spend_share": 0.05}`. Trường hợp hợp lệ dễ nhầm: Tài khoản tiết kiệm hoặc tài khoản chuyên thu không dùng chi tiêu. Hỗ trợ hiện tại: Cần bổ sung feature/công thức; không phải luật đã triển khai.
- **E04** — nguồn: `settled_history, observed_profile`; lịch tối thiểu: 30 ngày; tham số: `{"window_days": 30, "ratio": 3}`. Trường hợp hợp lệ dễ nhầm: Bán tài sản, khoản vay hoặc doanh thu thời vụ có giải thích. Hỗ trợ hiện tại: Cần bổ sung feature/công thức; không phải luật đã triển khai.
- **E05** — nguồn: `settled_history, directed_timeline`; lịch tối thiểu: 0 ngày; tham số: `{"window_seconds": 86400, "pair_seconds": 3600, "pairs": 3, "tolerance": 0.1}`. Trường hợp hợp lệ dễ nhầm: Thu hộ rồi chi hộ theo từng đơn. Hỗ trợ hiện tại: Cần bổ sung feature/công thức; không phải luật đã triển khai.
- **E06** — nguồn: `request_history`; lịch tối thiểu: 29 ngày; tham số: `{"short_seconds": 300, "long_seconds": 86400, "share": 0.8, "min_count": 10, "reference_days": 28}`. Trường hợp hợp lệ dễ nhầm: Giờ mở bán hoặc trả lương theo đợt. Hỗ trợ hiện tại: Cần bổ sung feature/công thức; không phải luật đã triển khai.

### F. Ngữ cảnh / thiết bị

| Mã | Tiêu chí / đơn vị | Thời điểm / loại | Công thức |
| --- | --- | --- | --- |
| F01 | Giao dịch vào giờ bất thường; transaction | Request / giám sát; behavior_signal | Giờ request hiện tại có prior fraction <max_fraction trong 30d và prior support>=min_requests; ban đêm không tự là rủi ro. |
| F02 | Đăng nhập từ thiết bị mới; transaction | Request / giám sát; behavior_signal | Có login thành công từ device chưa biết trước login; trong 1h tiếp theo request có amount/median>amount_ratio hoặc new_receiver. Account chưa có device history là insufficient_data. |
| F03 | Một thiết bị liên quan đến nhiều tài khoản; device_window | Request / giám sát; behavior_signal | distinct accounts có authenticated session trên cùng device trong 24h >=accounts; device fingerprint phải ổn định, có confidence/source. |
| F04 | Địa điểm đăng nhập thay đổi bất thường; account_window | Request / giám sát; behavior_signal | Hai login thành công liên tiếp có reliable geo; distance/travel_time >max_speed, loại vị trí không chắc chắn/VPN theo policy. Không suy ra geo chỉ từ IP thô. |
| F05 | Thay đổi bảo mật trước giao dịch lớn; transaction | Request / giám sát; behavior_signal | Security change event đã commit trước request <=1h và request amount/median>ratio hoặc new_receiver; ghi chính xác loại thay đổi, không dùng terminal auth_verified của request. |
| F06 | Nhiều tài khoản có dấu hiệu điều khiển chung; entity_group | Request / giám sát; behavior_signal | Nhóm >=accounts có shared authenticated device/session, request trong burst_seconds và có mẫu đối tác/amount liên quan; ghi evidence của từng quan hệ, không kết luận điều khiển chung. |
| F07 | Không thể liên hệ để xác minh; account_window | Xác minh; external_fact | VerificationAttempt có kết quả không liên hệ được theo protocol/version, known_at<=t; không suy ra từ thiếu metadata. |

- **F01** — nguồn: `request_history`; lịch tối thiểu: 30 ngày; tham số: `{"reference_days": 30, "max_fraction": 0.01, "min_requests": 20}`. Trường hợp hợp lệ dễ nhầm: Khách làm ca đêm hoặc có lịch hoạt động thay đổi. Hỗ trợ hiện tại: Có hour_prior_fraction toàn lịch sử, chưa window 30d.
- **F02** — nguồn: `request_history, settled_history, device_history, login_history`; lịch tối thiểu: 0 ngày; tham số: `{"login_to_request_seconds": 3600, "amount_ratio": 5}`. Trường hợp hợp lệ dễ nhầm: Khách đổi điện thoại rồi giao dịch hợp lệ. Hỗ trợ hiện tại: Có new_device tại transaction request, chưa có login events để xác nhận F02.
- **F03** — nguồn: `device_history, login_history`; lịch tối thiểu: 0 ngày; tham số: `{"window_seconds": 86400, "accounts": 3}`. Trường hợp hợp lệ dễ nhầm: Thiết bị gia đình hoặc người được ủy quyền quản lý nhiều tài khoản. Hỗ trợ hiện tại: Dataset hiện sinh ID device theo account, không có shared-device graph.
- **F04** — nguồn: `login_history, geo_history`; lịch tối thiểu: 0 ngày; tham số: `{"max_speed_kmh": 900}`. Trường hợp hợp lệ dễ nhầm: VPN, sai định vị hoặc mạng di động. Hỗ trợ hiện tại: Cần bổ sung feature/công thức; không phải luật đã triển khai.
- **F05** — nguồn: `security_changes, request_history, settled_history`; lịch tối thiểu: 0 ngày; tham số: `{"change_to_request_seconds": 3600, "amount_ratio": 5}`. Trường hợp hợp lệ dễ nhầm: Đổi mật khẩu hoặc số liên hệ theo quy trình hợp lệ. Hỗ trợ hiện tại: Cần bổ sung feature/công thức; không phải luật đã triển khai.
- **F06** — nguồn: `device_history, login_history, request_history, directed_timeline`; lịch tối thiểu: 0 ngày; tham số: `{"accounts": 3, "burst_seconds": 300}`. Trường hợp hợp lệ dễ nhầm: Nhóm khách dùng thiết bị được ủy quyền hoặc quầy dịch vụ. Hỗ trợ hiện tại: Cần bổ sung feature/công thức; không phải luật đã triển khai.
- **F07** — nguồn: `contact_checks`; lịch tối thiểu: 0 ngày; tham số: `{}`. Trường hợp hợp lệ dễ nhầm: Khách tạm thời không nghe máy. Hỗ trợ hiện tại: Cần bổ sung feature/công thức; không phải luật đã triển khai.

### G. Định danh / hồ sơ

| Mã | Tiêu chí / đơn vị | Thời điểm / loại | Công thức |
| --- | --- | --- | --- |
| G01 | Thông tin tài khoản không khớp dữ liệu định danh; customer_or_account | Xác minh; external_fact | IdentityCheck cá nhân trả mismatch với nguồn/version và known_at<=t. |
| G02 | Thông tin tổ chức không khớp; customer_or_account | Xác minh; external_fact | BusinessRegistryCheck tổ chức trả mismatch, nguồn/version và known_at<=t. |
| G03 | Thông tin người đại diện không khớp; customer_or_account | Xác minh; external_fact | RepresentativeCheck trả mismatch, định danh người đại diện và known_at<=t. |
| G04 | Tài khoản nằm trong danh sách mua bán; customer_or_account | Xác minh; external_fact | Evidence tài khoản được rao bán đã xác minh nguồn, timestamp, độ tin cậy và ánh xạ đúng account. |
| G05 | Chủ tài khoản thuộc danh sách cảnh báo; customer_or_account | Xác minh; external_fact | Customer/account/representative watchlist active as-of t, có source/version và thời điểm hệ thống biết. |
| G06 | Giao dịch không phù hợp hồ sơ; account_window | Rà soát định kỳ; contextual_review | Đối chiếu amount/frequency/counterparties theo 30d với hồ sơ thu nhập/nghề nghiệp/doanh thu có hiệu lực trước t; cần reviewer outcome để kết luận không phù hợp. |
| G07 | Mối quan hệ tài khoản không rõ ràng; account_window | Xác minh; contextual_review | Reviewer xác minh mục đích/nguồn tiền/quan hệ đối tác; chỉ ghi outcome đã biết tại t, chưa xác minh là unknown. |

- **G01** — nguồn: `identity_checks`; lịch tối thiểu: 0 ngày; tham số: `{}`. Trường hợp hợp lệ dễ nhầm: Sai khác do dữ liệu cũ hoặc lỗi nhập liệu. Hỗ trợ hiện tại: Cần bổ sung feature/công thức; không phải luật đã triển khai.
- **G02** — nguồn: `identity_checks`; lịch tối thiểu: 0 ngày; tham số: `{}`. Trường hợp hợp lệ dễ nhầm: Thông tin doanh nghiệp đang cập nhật. Hỗ trợ hiện tại: Cần bổ sung feature/công thức; không phải luật đã triển khai.
- **G03** — nguồn: `identity_checks`; lịch tối thiểu: 0 ngày; tham số: `{}`. Trường hợp hợp lệ dễ nhầm: Thay đổi người đại diện hợp lệ chưa đồng bộ. Hỗ trợ hiện tại: Cần bổ sung feature/công thức; không phải luật đã triển khai.
- **G04** — nguồn: `watchlist_asof`; lịch tối thiểu: 0 ngày; tham số: `{}`. Trường hợp hợp lệ dễ nhầm: Tin giả hoặc nhầm số tài khoản. Hỗ trợ hiện tại: Cần bổ sung feature/công thức; không phải luật đã triển khai.
- **G05** — nguồn: `watchlist_asof`; lịch tối thiểu: 0 ngày; tham số: `{}`. Trường hợp hợp lệ dễ nhầm: Nhầm định danh hoặc bản ghi hết hiệu lực. Hỗ trợ hiện tại: Cần bổ sung feature/công thức; không phải luật đã triển khai.
- **G06** — nguồn: `observed_profile, request_history, settled_history`; lịch tối thiểu: 30 ngày; tham số: `{}`. Trường hợp hợp lệ dễ nhầm: Khách thay đổi công việc hoặc mở hoạt động kinh doanh. Hỗ trợ hiện tại: Cần bổ sung feature/công thức; không phải luật đã triển khai.
- **G07** — nguồn: `purpose_evidence`; lịch tối thiểu: 0 ngày; tham số: `{}`. Trường hợp hợp lệ dễ nhầm: Đối tác mới hợp lệ chưa cung cấp chứng từ. Hỗ trợ hiện tại: Cần bổ sung feature/công thức; không phải luật đã triển khai.

### H. Mẫu hình / mục đích

| Mã | Tiêu chí / đơn vị | Thời điểm / loại | Công thức |
| --- | --- | --- | --- |
| H01 | Nhiều giao dịch có mẫu hình gần giống nhau; account_window | Request / giám sát; behavior_signal | Trong 1h có >=count requests với amount sai lệch<=tolerance quanh amount đại diện và cùng nhóm đối tác; đây là mẫu lặp, không kết luận giả tạo. |
| H02 | Giao dịch lặp lại không rõ mục đích kinh tế; account_window | Xác minh; contextual_review | Có D05/D06 và bằng chứng reviewer rằng không có nghĩa vụ/mục đích tương ứng; graph pattern đơn lẻ không đủ kết luận. |
| H03 | Hoạt động nhằm nhận thưởng bất thường; account_window | Xác minh; contextual_review | Reward context có program_id/version; hoạt động đủ điều kiện thưởng tăng bất thường và có kết quả xác minh giao dịch không thực chất. |
| H04 | Nhiều giao dịch bị hủy hoặc điều chỉnh; account_window | Sau settled / giám sát; behavior_signal | Trong 24h: failed+cancelled terminal count / requests đã có terminal >=ratio, số terminal>=min_count và vượt baseline 28 ngày. Refund/adjustment là nhánh riêng, cần refund events; pending không là failure. |
| H05 | Hành vi không phù hợp mục đích khai báo; account_window | Xác minh; contextual_review | D07 kết hợp hồ sơ mục đích tài khoản và reviewer xác nhận không phù hợp; không gán gian lận từ fan-in/fan-out đơn lẻ. |

- **H01** — nguồn: `request_history, directed_timeline`; lịch tối thiểu: 0 ngày; tham số: `{"window_seconds": 3600, "count": 5, "tolerance": 0.01}`. Trường hợp hợp lệ dễ nhầm: Thanh toán theo bảng giá hoặc hóa đơn cùng mệnh giá. Hỗ trợ hiện tại: Cần bổ sung feature/công thức; không phải luật đã triển khai.
- **H02** — nguồn: `settled_history, directed_timeline, purpose_evidence`; lịch tối thiểu: 0 ngày; tham số: `{}`. Trường hợp hợp lệ dễ nhầm: Luân chuyển quỹ hoặc hoàn trả hợp lệ. Hỗ trợ hiện tại: Cần bổ sung feature/công thức; không phải luật đã triển khai.
- **H03** — nguồn: `reward_context, request_history, purpose_evidence`; lịch tối thiểu: 0 ngày; tham số: `{}`. Trường hợp hợp lệ dễ nhầm: Chiến dịch kích cầu hợp lệ. Hỗ trợ hiện tại: Cần bổ sung feature/công thức; không phải luật đã triển khai.
- **H04** — nguồn: `request_history, terminal_history, refund_history`; lịch tối thiểu: 29 ngày; tham số: `{"window_seconds": 86400, "ratio": 0.5, "min_count": 10, "reference_days": 28}`. Trường hợp hợp lệ dễ nhầm: Sự cố hạ tầng, khách hủy hoặc hoàn tiền hợp lệ. Hỗ trợ hiện tại: Có failed/cancelled terminals; chưa có refund/adjustment events. Chỉ có thể xét nhánh failure riêng.
- **H05** — nguồn: `settled_history, directed_timeline, observed_profile, purpose_evidence`; lịch tối thiểu: 0 ngày; tham số: `{}`. Trường hợp hợp lệ dễ nhầm: Tài khoản thu hộ được phép hoặc thay đổi mục đích hợp lệ. Hỗ trợ hiện tại: Cần bổ sung feature/công thức; không phải luật đã triển khai.

<!-- CRITERIA_CATALOG_END -->

## 5. Ví dụ có thể kiểm chứng

Tại 09:00–09:20, tài khoản nhận **settled** 35 triệu từ 8 account khác nhau. Tại 09:40, tài khoản request chuyển 33 triệu cho một người nhận mới; giao dịch hoàn tất lúc 09:41. Sổ cái đầy đủ cho biết số dư trước đợt nhận là 0, không phí và không có bút toán khác. Baseline 28 cửa sổ ngày trước có median incoming 12 triệu/ngày, đủ support. Tham số A02 minimum_amount=10 triệu, C01 minimum_inflow=10 triệu được cấu hình trước khi đánh giá.

| Tiêu chí | Lúc request 09:40 | Sau hoàn tất 09:41 |
| --- | --- | --- |
| A02 | Không khớp: 35/12≈2,92 <3, dù tổng tiền lớn | Không khớp với các giá trị này |
| B03 | Tùy cutoff 30m: cần đếm chính xác những khoản còn trong [09:10,09:40); không mặc định cả 8 nguồn | Tính lại cửa sổ theo timestamp |
| C01 | Settled outflow chưa tăng; projected ratio=33/35≈94,29%, gắn rõ projected | Settled ratio≈94,29% trong 60m, khớp nhánh đã cấu hình |
| D03 | Người nhận mới của request chưa chứng minh >=10 đối tác mới/24h | Chỉ 1 người nhận mới vẫn chưa đạt ngưỡng count |
| C02/C04 | insufficient_data: ngày chưa kết thúc | Vẫn chờ cuối ngày; C04 còn cần số ngày lặp |
| D01 | insufficient_data nếu chưa có watchlist as-of | Không suy ra từ nhãn kịch bản |

Kết quả cảnh báo có thể đáng xem xét nhưng không tự chứng minh gian lận. Có thể dùng cùng cấu trúc dòng tiền cho kịch bản thu hộ hợp lệ, với hồ sơ và quan hệ đối tác giải thích được.

## 6. Áp dụng với dataset và GATv2 hiện tại

Ưu tiên B03/B04, C01/C03/C06 và D03/D04, sau khi thêm công thức/feature tương ứng. Graph model hiện bidirectional và deduplicate cặp node, nên D05/D06 cần tính từ directed event timeline, không suy ra chuỗi/thứ tự từ adjacency đã làm đơn giản. Device hiện sinh riêng theo account, chưa có shared-device hoặc login/security history; không khẳng định đáp ứng F02/F03/F05/F06.

EDA chạy trên dataset hiện có để mô tả coverage, trạng thái thiếu source, lịch sử tối đa có thể quan sát, phân bố feature, redundancy và sensitivity validation của checkpoint đúng schema. EDA không sinh lại nhãn, không đo F1 của bộ tiêu chí chưa triển khai và không đổi model/threshold theo test. Profile/generator, feature và guard là các bước triển khai sau đặc tả này.

Chấp nhận một tiêu chí mới khi có source contract, causal tests cho boundary/same-time/dedup, bằng chứng mẫu khớp/không khớp/chưa đủ dữ liệu, trường hợp bình thường dễ nhầm và protocol đánh giá được chốt trước. Khi mở rộng generator, dùng nhiều seed và holdout kịch bản mới; giữ scaler fit train và chọn checkpoint/threshold bằng validation.

## 7. Tài liệu tham khảo

- Thông tư 25/2025/TT-NHNN — văn bản sửa đổi, bổ sung một số điều của quy định về mở và sử dụng tài khoản thanh toán. Tra cứu văn bản: https://vbpl.vn/TW/Pages/vbpq-toanvan.aspx?ItemID=181766

- Thư viện Pháp luật — “Từ năm 2026, một số giao dịch chuyển khoản có dấu hiệu bất thường có thể được ngân hàng rà soát”: https://thuvienphapluat.vn/ma-so-thue/phap-luat-thue/tu-nam-2026-mot-so-giao-dich-chuyen-khoan-co-dau-hieu-bat-thuong-co-the-duoc-ngan-hang-ra-soat-xac--891362-225346.html02

- XT Support — “Quy định xử lý hành vi giao dịch bất thường”: https://xtsupport.zendesk.com/hc/vi/articles/45471864599193--Quy-%C4%91%E1%BB%8Bnh-x%E1%BB%AD-l%C3%BD-h%C3%A0nh-vi-giao-d%E1%BB%8Bch-b%E1%BA%A5t-th%C6%B0%E1%BB%9Dng

- DNSE — “Chỉ tiêu bất thường”: https://www.dnse.com.vn/hoc/chi-tieu-bat-thuong

- Tài liệu Quốc hội — “Những tiêu chí phân loại hành vi giao dịch bất hợp pháp trên thị trường chứng khoán tập trung”: http://tailieu.quochoi.vn:8080/index.php/tin-tuc/tin-tuc-ho-tro-boi-duong/item/2811-nh-ng-tieu-chi-phan-lo-i-hanh-vi-giao-d-ch-b-t-h-p-phap-tren-th-tru-ng-ch-ng-khoan-t-p-trung
