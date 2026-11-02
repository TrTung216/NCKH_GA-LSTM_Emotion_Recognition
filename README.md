# UIT-VSMEC: dữ liệu huấn luyện và đánh giá

**Cập nhật 28/10/2026:** đã chạy LSTM, xLSTM (mLSTM-only), GA+LSTM và GA+xLSTM, mỗi cấu hình cuối đánh giá trên 3 seed. Xem [thực nghiệm cấu hình cố định](EXPERIMENTS.md), [hướng dẫn GA](GA_EXPERIMENTS.md) và [báo cáo so sánh bốn phương pháp](results/ga_lstm_xlstm_comparison/REPORT_VI.md). GA dùng 18 trial mỗi kiến trúc; đây là nghiên cứu mở rộng thăm dò trên cùng test đã được xem ở lần trước. `RESEARCH_NOTE.md` là tài liệu thiết kế tại ngày 25/10; các mục “chưa chạy” trong tài liệu đó phản ánh thời điểm viết, và đặc tả GA 32 trial trong tài liệu đó chưa được chạy nguyên vẹn.

Tải lần đầu ngày 04/10/2026. Ngày 25/10/2026, tải lại ba URL nguồn và xác nhận SHA-256 khớp hoàn toàn. Nguồn: thư mục được trang UIT NLP liên kết ở lần chuẩn bị trước; hiện trang này chuyển hướng sang trang chủ UIT, các URL file trực tiếp vẫn tải được.

Đây là bài toán phân loại **một nhãn trong bảy lớp** cho mỗi bình luận tiếng Việt. Gói này giữ nguyên văn bản, nhãn, thứ tự dòng và tập chia của ba workbook chính thức. Các file có hậu tố `nor_811` là bản được nguồn phát hành dưới tên normalized; không gọi chúng là bình luận thô chưa tiền xử lý.

## Các tập dữ liệu

| Tập | Số mẫu | Mục đích |
|---|---:|---|
| Train | 5.548 | Học trọng số, xây vocabulary hoặc fit TF-IDF |
| Validation | 686 | Chọn siêu tham số, epoch, cách tiền xử lý |
| Test | 693 | Đánh giá cuối khi đã khóa phương pháp |
| Tổng | 6.927 | 6.921 chuỗi văn bản phân biệt tuyệt đối |

Các file:

- `original/`: ba file XLSX tải trực tiếp, giữ nguyên byte.
- `data/train.csv`, `data/validation.csv`, `data/test.csv`: bản UTF-8 có BOM, dùng được với pandas và Excel.
- `data/*.jsonl`: nội dung tương đương, mỗi dòng một đối tượng JSON.
- `label_mapping.json`: mã nhãn thống nhất do gói này quy định.
- `sources.json`: URL, ngày tải và SHA-256 của file nguồn.
- `audit/`: phân bố nhãn, dữ liệu trùng, nhãn mâu thuẫn và báo cáo chất lượng.
- `scripts/prepare_data.py`: tái tạo dữ liệu và thống kê từ workbook.
- `scripts/evaluate.py`: đánh giá dự đoán có kiểm tra ID.
- `RESEARCH_NOTE.md`: phân tích bài toán, thiết kế GA–xLSTM, đối chứng, tài nguyên và lộ trình nghiên cứu.
- `data_variants/train_no_validation_overlap.*`: biến thể train 5.547 mẫu, bỏ đúng một dòng trùng văn bản validation, giữ nguyên các tập gốc.
- `configs/ga_xlstm_research_plan.json`: đặc tả thí nghiệm GA–xLSTM; chưa phải mã huấn luyện chạy được.
- `scripts/prepare_training_variant.py`: tái tạo biến thể train.
- `scripts/baseline_tfidf.py`: chạy baseline TF-IDF + LinearSVC trên validation với cấu hình cố định.
- `results/baseline_validation/`: dự đoán, điểm từng lớp và metadata của lần chạy baseline thực tế.
- `scripts/verify_package.py`: kiểm tra hash của các file trong gói.

## Cấu trúc một mẫu

| Cột | Ý nghĩa |
|---|---|
| `sample_id` | ID duy nhất được tạo từ tập chia và vị trí dòng, ví dụ `train_00000` |
| `source_row` | Số dòng Excel, tính cả dòng tiêu đề |
| `source_index` | Cột chỉ số được giữ lại từ workbook; không duy nhất, không dùng làm khóa |
| `text` | Văn bản nguyên văn từ cột Sentence |
| `label` | Tên nhãn nguyên văn từ cột Emotion |
| `label_id` | Số nguyên từ 0 đến 6 theo bảng dưới |

| ID | Nhãn | Nghĩa |
|---:|---|---|
| 0 | Anger | Giận dữ |
| 1 | Disgust | Ghê tởm, chán ghét |
| 2 | Enjoyment | Vui thích |
| 3 | Fear | Sợ hãi |
| 4 | Other | Ngoài sáu nhóm hoặc không biểu lộ cảm xúc |
| 5 | Sadness | Buồn bã |
| 6 | Surprise | Ngạc nhiên |

Thứ tự ID này là quy ước của gói, không phải mã số chính thức và không nhất thiết khớp checkpoint tải từ nơi khác.

## Đọc dữ liệu

Chạy từ thư mục `UIT_VSMEC` sau khi giải nén:

```bash
python -m pip install -r requirements.txt
```

```python
import pandas as pd

train = pd.read_csv('data/train.csv', encoding='utf-8-sig')
validation = pd.read_csv('data/validation.csv', encoding='utf-8-sig')
test = pd.read_csv('data/test.csv', encoding='utf-8-sig')

texts = train['text'].tolist()
labels = train['label_id'].tolist()
# Với PyTorch: logits có shape [batch_size, 7]; dùng CrossEntropyLoss.
```

Không đưa các cột ID, source_row hoặc source_index vào đặc trưng mô hình. Không fit tokenizer/vocabulary/TF-IDF trên validation hay test nếu học chúng từ bộ dữ liệu này.

## Đánh giá

Xuất dự đoán ra CSV với hai cột `sample_id,pred_label`. `pred_label` dùng tên nhãn chính xác, không dùng số ID. File phải có đúng một dòng cho mỗi mẫu cần đánh giá. Script sẽ ghép theo ID nên thứ tự dự đoán có thể khác thứ tự dữ liệu.

```bash
python scripts/evaluate.py --gold data/validation.csv --pred predictions_validation.csv --output results/validation_metrics.json
python scripts/evaluate.py --gold data/test.csv --pred predictions_test.csv --output results/test_metrics.json
```

Kết quả gồm Accuracy, Macro-F1, Weighted-F1, Precision/Recall/F1 từng lớp, support và confusion matrix (dòng: nhãn thật; cột: nhãn dự đoán). Các chỉ số có thang 0–1. Chọn mô hình theo Macro-F1 trên validation. Test chỉ dùng sau khi khóa thiết kế. Baseline TF-IDF + LinearSVC có kết quả validation. Thực nghiệm LSTM/xLSTM ngày 28/10 có kết quả validation và test cho cả 3 seed, lưu riêng trong `results/lstm_xlstm_comparison/`. Thực nghiệm GA+LSTM/GA+xLSTM lưu trong `results/ga_lstm_xlstm_comparison/`, đã chấm lại độc lập 56 bộ dự đoán; chưa chạy mô hình tiền huấn luyện.

## Bắt đầu từ baseline đã kiểm tra

```bash
python scripts/verify_package.py
python scripts/prepare_training_variant.py
python scripts/baseline_tfidf.py
```

Baseline dùng train 5.547 mẫu và validation 686 mẫu, đạt Accuracy 57,58%, Macro-F1 53,19%, Weighted-F1 57,55%. Cấu hình được đặt trước khi đánh giá, chưa tìm siêu tham số. Kết quả không phải test score và không phải kết quả GA–xLSTM. Phiên bản thư viện thực chạy ở `requirements-baseline-checked.txt` và `results/baseline_validation/run_metadata.json`.

Mặc định của script baseline là biến thể train loại một điểm giao nhau với validation. Dữ liệu trong `data/` vẫn là bản phát hành nguyên gốc. Dùng cùng một biến thể train cho tất cả mô hình trong một bảng so sánh. Sau khi chạy lại baseline, các file kết quả và thời gian sẽ đổi nên manifest hash ban đầu không còn áp dụng cho phần kết quả đã ghi đè.

## Các điểm đã kiểm tra

- Không có văn bản/nhãn bị thiếu hoặc văn bản chỉ chứa khoảng trắng.
- Cả ba tập đều có đủ bảy lớp.
- Train có 4 dòng trùng văn bản tính sau lần xuất hiện đầu; test có 1; validation có 0.
- Có 1 chuỗi trùng giữa train và validation; không có trùng chuỗi chính xác train–test hay validation–test. Kết quả giống nhau khi chỉ chuẩn hóa NFC, chữ thường và khoảng trắng.
- Có 1 chuỗi văn bản trong train mang hai nhãn khác nhau. Chi tiết nằm trong `audit/conflicting_labels.csv`.
- Chưa kiểm tra trùng ngữ nghĩa, bình luận cùng bài đăng hoặc quan hệ hội thoại. Không suy ra rằng đã loại hết mọi khả năng rò rỉ.
- Tổng số dòng mỗi tập khớp bài gốc, nhưng phân bố từng lớp theo tập khác Table 7 trong bản arXiv v2. Dùng thống kê từ file thực tế trong `audit/label_distribution.csv` khi báo cáo.

Không tự động xóa hay sửa nhãn. Khi cần đánh giá độ nhạy, tạo biến thể riêng: xử lý mẫu trùng trong train theo quy tắc được chốt trước, xử lý nhãn mâu thuẫn theo hướng dẫn gán nhãn, và loại khỏi train các bản sao của validation. Giữ test gốc cho đối chiếu; nếu thêm kết quả test loại trùng thì ghi rõ số mẫu và không gọi đó là kết quả trên toàn bộ test chính thức.

## Nguồn và sử dụng

- [Trang UIT NLP](https://nlp.uit.edu.vn/datasets)
- [Thư mục tải chính thức](https://drive.google.com/drive/folders/1HooABJyrddVGzll7fgkJ6VzkG_XuWfRu)
- [Ho et al., Emotion Recognition for Vietnamese Social Media Text](https://arxiv.org/abs/1911.09339)
- [Bản xuất bản PACLING 2019, online 2020](https://doi.org/10.1007/978-981-15-6168-9_27)

Bài gốc giới thiệu corpus cho mục đích nghiên cứu. Các workbook tải về không kèm một file giấy phép riêng; gói này không tự gán giấy phép MIT hay quyền thương mại cho dữ liệu. Khi công bố kết quả, dẫn bài gốc và ghi rõ phiên bản workbook/ngày tải. Tài liệu và script mới chỉ là phần hỗ trợ sử dụng dữ liệu.
