# Chạy thực nghiệm LSTM và xLSTM

Thực nghiệm ngày 28/10/2026 so sánh LSTM và xLSTM (hai block mLSTM) trên
UIT-VSMEC, với 3 seed cố định và cùng quy trình huấn luyện. Không dùng GA.

Kết quả và biểu đồ: [REPORT_VI.md](results/lstm_xlstm_comparison/REPORT_VI.md).
Cấu hình: [lstm_xlstm_comparison.json](configs/lstm_xlstm_comparison.json).

Thực nghiệm mở rộng dùng GA để tìm siêu tham số cho cả hai kiến trúc:
[GA_EXPERIMENTS.md](GA_EXPERIMENTS.md). Các kết quả cấu hình cố định tại đây
được giữ nguyên làm đối chứng.

## Chạy lại

Python 3.11, PyTorch có CUDA nếu dùng GPU. Phiên bản thực chạy được ghi trong
`results/lstm_xlstm_comparison/environment.json`. Các thư viện bổ sung hiện
được cài vào `.experiment_deps/`, không thay đổi PyTorch hệ thống.

Nếu tạo môi trường mới, cài PyTorch phù hợp phần cứng rồi chạy:

```powershell
python -m pip install -r requirements-experiments.txt
python scripts/check_sequence_models.py
python scripts/compare_lstm_xlstm.py --outdir results/new_comparison
python scripts/report_lstm_xlstm.py --outdir results/new_comparison
python scripts/verify_sequence_experiment.py --outdir results/new_comparison
```

Giữ thư mục `vendor/` đi kèm: đây là mã xLSTM chính thức tại commit đã pin,
với một thay đổi lazy-import được mô tả trong [vendor/README.md](vendor/README.md).
Không cần build kernel sLSTM CUDA để chạy thí nghiệm mLSTM này.
`.experiment_deps/` được ưu tiên import nếu tồn tại; để dùng thư viện trong
môi trường mới, không đưa thư mục này vào bản sao dự án.

Script mặc định tự chọn CUDA khi khả dụng; có thể dùng `--device cpu`.
Thêm `--validation-only` để chỉ train và đánh giá validation mà không đọc test.
Thư mục output phải mới hoặc rỗng để tránh ghi đè kết quả.

## Ý nghĩa các file

- `locked_config.json`: cấu hình lưu trước huấn luyện.
- `tokenizer.model`, `tokenizer.vocab`: SentencePiece chỉ học trên train.
- `environment.json`: phần cứng, phiên bản thư viện, hash dữ liệu/mã nguồn/tokenizer,
  thống kê cắt ngắn và token không biết.
- `training_complete.json`: thời điểm hoàn tất toàn bộ train, trước đánh giá test.
- `<model>_seed<seed>/best.pt`: trọng số checkpoint chọn bằng validation, cấu hình,
  loại mô hình, seed, kích thước vocabulary và epoch đã chọn.
- `history.csv`: loss train và điểm validation qua từng epoch.
- `metrics_*.json`, `predictions_*.csv`: điểm từng lớp, ma trận nhầm lẫn, dự đoán theo ID.
- `summary_runs.csv`, `summary_mean_std.csv`: kết quả từng seed và trung bình ± độ lệch chuẩn.
- `REPORT_VI.md` và các PNG: báo cáo tiếng Việt và biểu đồ.

Không lấy seed tốt nhất để đại diện cho mô hình. Không điều chỉnh cấu hình từ
kết quả test. Ba seed giúp quan sát dao động huấn luyện nhưng chưa đủ để kết luận
phổ quát về kiến trúc; xem các giới hạn trong báo cáo.
