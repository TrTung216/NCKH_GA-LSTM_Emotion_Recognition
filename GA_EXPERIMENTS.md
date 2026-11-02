# Thực nghiệm GA + LSTM và GA + xLSTM

GA tìm siêu tham số cho từng kiến trúc; AdamW vẫn học trọng số mạng.
Thực nghiệm này tiếp nối kết quả cấu hình cố định, sử dụng đúng tokenizer,
dữ liệu và mã huấn luyện của lần trước.

Đây là thực nghiệm 18 trial mỗi kiến trúc theo cấu hình mới bên dưới,
không phải chạy nguyên đặc tả nghiên cứu 32 trial trong
`configs/ga_xlstm_research_plan.json` của ngày 25/10.

- [Cấu hình GA đã cố định](configs/ga_lstm_xlstm_comparison.json)
- [Báo cáo, bảng kết quả và biểu đồ](results/ga_lstm_xlstm_comparison/REPORT_VI.md)
- [Thực nghiệm cấu hình cố định](EXPERIMENTS.md)

## Quy trình

1. Mỗi kiến trúc có quần thể 6, 3 thế hệ sinh thêm, giữ 2 elite: tổng 18
   cấu hình duy nhất. Quần thể ban đầu giống nhau, bao gồm cấu hình cũ.
2. Tìm embedding_dim, num_layers, dropout, learning_rate, weight_decay và
   class_weight_alpha. Mỗi trial train 5 epoch, fitness là Macro-F1 validation tốt nhất.
3. Top 2 cấu hình được train lại với seed 17, 29 (tối đa 20 epoch, patience 5).
   Chọn cấu hình có Macro-F1 validation trung bình cao nhất.
4. Khóa cấu hình cho cả hai mô hình, train từ đầu với seed 42, 123, 2026
   (tối đa 20 epoch, patience 5), chọn checkpoint bằng validation.
5. Sau khi cả sáu lần train cuối xong mới đọc test để đánh giá. Báo cáo tất cả seed.

Chạy một seed GA (71) cho mỗi kiến trúc. Ba seed cuối phản ánh dao động train,
không đo dao động giữa các lần chạy GA độc lập. Chưa có đối chứng random search.
Seed cuối được giữ giống baseline để so sánh theo cặp; seed 42 trùng seed train
proxy, vì vậy không coi cả ba seed cuối là hoàn toàn độc lập với quá trình chọn cấu hình.
Test đã được công bố ở thực nghiệm trước, nên đây là nghiên cứu mở rộng thăm dò,
không phải xác nhận trên test mới. Không dùng điểm test để điều khiển GA.

## Chạy lại

Dùng cùng môi trường trong `requirements-experiments-checked.txt`, thư mục
`vendor/` và kết quả baseline ở `results/lstm_xlstm_comparison/`.

```powershell
python scripts/check_ga_search.py
python scripts/run_ga_comparison.py --outdir results/new_ga_run
python scripts/report_ga_comparison.py --outdir results/new_ga_run
python scripts/verify_ga_comparison.py --outdir results/new_ga_run
```

Để tiếp tục một lần chạy bị gián đoạn:

```powershell
python scripts/run_ga_comparison.py --outdir results/new_ga_run --resume
```

Resume yêu cầu cấu hình, dữ liệu, tokenizer, mã nguồn và phiên bản thư viện
khớp với bản đã lưu. Các trial hoàn tất được dùng lại. Run dở được giữ trong
`interrupted/` rồi huấn luyện lại từ đầu với cùng seed, không tiếp tục optimizer.
Thư mục kết quả có dữ liệu sẽ không bị ghi đè nếu thiếu `--resume`.

Không sửa cấu hình khi resume. Nếu muốn thay không gian tìm kiếm hay ngân sách,
tạo file config mới và dùng thư mục kết quả mới với `--config`.

## File kết quả

- `locked_protocol.json`: cấu hình GA, cấu hình train gốc, hash dữ liệu và mã nguồn.
- `code/`: bản chụp ba file mã thực thi tại thời điểm bắt đầu.
- `search/<model>/search_state.json`: fitness, cấu hình, bố mẹ, gene đột biến,
  quần thể từng thế hệ, số đề xuất trùng bị loại.
- `search/<model>/trial_XXX/`: checkpoint, lịch sử, dự đoán validation, cấu hình
  hiệu lực và log của từng trial. Trial ID bắt đầu từ 0.
- `rerank/`, `reranking_<model>.json`: kết quả chạy lại các ứng viên.
- `selected_configs.json`: cấu hình được chọn và thời điểm khóa, trước train cuối/test.
- `final/`: checkpoint, lịch sử, dự đoán validation/test của sáu run cuối.
- `summary_runs.csv`: kết quả từng seed của hai mô hình có GA.
- `comparison_runs.csv`, `comparison_mean_std.csv`: cả bốn phương pháp.
- `costs.csv`: thời gian search, chạy lại chọn cấu hình, train cuối.
- `verification/verification.json`: kiểm tra hash, ngân sách, elitism, chọn cấu hình,
  chọn epoch và chấm lại dự đoán bằng `scripts/evaluate.py`.

Thời gian GPU không bằng nhau dù số trial giống nhau. xLSTM dùng backend mLSTM
thuần PyTorch; không coi đây là so sánh hiệu năng kernel được tối ưu ngang nhau.
