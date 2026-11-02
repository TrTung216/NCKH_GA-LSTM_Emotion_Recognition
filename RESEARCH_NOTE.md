# Nhận diện cảm xúc tiếng Việt và nghiên cứu tối ưu xLSTM bằng GA

Cập nhật: **25/10/2026**. Phạm vi: cảm xúc biểu đạt trong văn bản tiếng Việt, trước hết là bình luận mạng xã hội. Các cấu hình GA–xLSTM dưới đây là **đề xuất nghiên cứu**, chưa phải kết quả huấn luyện.

## 1. Kết quả đã hoàn thành

| Hạng mục | Trạng thái |
|---|---|
| UIT-VSMEC | Đã có đủ train, validation, test; tải lại ba nguồn ngày 25/10 và đối chiếu SHA-256 khớp hoàn toàn |
| Định dạng | XLSX gốc, CSV UTF-8, JSONL, ánh xạ nhãn và ID từng mẫu |
| Kiểm tra dữ liệu | Đã đối chiếu XLSX–CSV–JSONL, số mẫu, nhãn, trùng lặp và nhãn mâu thuẫn |
| Biến thể train cho tìm kiếm | Đã tạo bản bỏ một dòng trùng văn bản với validation; còn 5.547 mẫu |
| Baseline nhẹ | Đã chạy một cấu hình cố định TF-IDF + Linear SVM, đánh giá trên 686 mẫu validation |
| GA–xLSTM | Đã xây dựng đặc tả kiến trúc, chromosome, fitness, ngân sách và đối chứng; chưa chạy |
| Đánh giá mô hình trên test | Chưa thực hiện; test hiện mới được kiểm tra chất lượng dữ liệu |
| ViGoEmotions | Đã kiểm tra bài báo và điều kiện truy cập; chưa tải dữ liệu có kiểm soát truy cập |

Đề tài đề xuất: **“Tối ưu xLSTM bằng thuật toán di truyền cho nhận diện cảm xúc tiếng Việt dưới ngân sách tính toán giới hạn.”**

## 2. Bài toán cần giải là gì?

Đầu vào là một bình luận tiếng Việt; đầu ra là cảm xúc được biểu đạt trong bình luận. Với UIT-VSMEC, mỗi mẫu mang **một nhãn trong bảy lớp**. Sentiment analysis thường phân loại sắc thái tích cực/tiêu cực/trung lập; emotion recognition phân biệt cụ thể hơn, ví dụ buồn, sợ và giận. [1]

| Nhãn | Ý nghĩa | Ví dụ tự tạo để giải thích |
|---|---|---|
| Anger | Tức giận | “Hẹn ba lần rồi vẫn không đến, tức thật!” |
| Disgust | Ghê tởm, chán ghét | “Mùi này làm mình muốn ói.” |
| Enjoyment | Vui thích | “Cuối cùng cũng đỗ rồi, vui quá!” |
| Fear | Sợ hãi | “Đi qua đoạn này một mình sợ quá.” |
| Other | Ngoài sáu nhóm hoặc không biểu lộ rõ cảm xúc | “Buổi học bắt đầu lúc tám giờ.” |
| Sadness | Buồn bã | “Mất hết ảnh kỷ niệm rồi, buồn thật.” |
| Surprise | Ngạc nhiên | “Ủa, hôm nay đã công bố kết quả rồi sao?” |

Các ví dụ trên không lấy từ corpus. **Không tự đổi Other thành Neutral.** Một câu ngoài đời có thể mang nhiều cảm xúc, nhưng nhãn của UIT-VSMEC chỉ hỗ trợ bài toán đơn nhãn. Mô hình phân loại biểu hiện ngôn ngữ; kết quả không xác nhận trạng thái tâm lý thực của người viết.

Khó khăn cần kiểm chứng trong phân tích lỗi: teencode, thiếu dấu, từ kéo dài, emoji, phủ định, mỉa mai, câu ngắn thiếu ngữ cảnh và ranh giới giữa các cảm xúc gần nhau. Ví dụ “không vui” phải khác “vui”; không xóa phủ định hoặc stopword hàng loạt. Các nhóm như mỉa mai cần được người đọc xác nhận, không chỉ suy ra từ một từ khóa.

## 3. Dữ liệu sử dụng và những điểm phải giữ đúng

### 3.1. UIT-VSMEC: bộ chính đã tải

Tổng **6.927 mẫu**, trong đó 6.921 chuỗi văn bản phân biệt tuyệt đối. Phân bố dưới đây được tính từ chính ba workbook phát hành, không suy ra từ tỉ lệ chia tập:

| Nhãn | Train | Validation | Test | Tổng |
|---|---:|---:|---:|---:|
| Anger | 391 | 49 | 40 | 480 |
| Disgust | 1.071 | 135 | 132 | 1.338 |
| Enjoyment | 1.558 | 214 | 193 | 1.965 |
| Fear | 318 | 31 | 46 | 395 |
| Other | 1.021 | 141 | 129 | 1.291 |
| Sadness | 947 | 86 | 116 | 1.149 |
| Surprise | 242 | 30 | 37 | 309 |
| **Tổng** | **5.548** | **686** | **693** | **6.927** |

Train học trọng số và tokenizer; validation chọn cấu hình, epoch và cách xử lý; test dùng để báo cáo cuối sau khi khóa phương pháp. Các ID 0–6 trong `label_mapping.json` là quy ước của gói, không phải mã số nhãn chính thức của mọi checkpoint.

Các workbook `train_nor_811.xlsx`, `valid_nor_811.xlsx`, `test_nor_811.xlsx` đã được tải từ thư mục nguồn ở lần chuẩn bị trước. Ngày 25/10/2026, tải lại cả ba URL và kiểm tra hash: **byte giống hoàn toàn**. `sources.json` ghi URL và hash; `audit/source_reverification_2026-10-25.json` ghi kết quả kiểm tra lại. Tại lần kiểm tra này, trang `nlp.uit.edu.vn/datasets` chuyển hướng sang trang chủ UIT, nhưng các URL tải trực tiếp vẫn trả đúng file. [2]

Hậu tố `nor_811` chỉ bản phát hành đã chuẩn hóa. Không gọi thí nghiệm trên các file này là “raw so với normalized” nếu chưa có bản raw được xác nhận và ghép từng mẫu. Tổng nhãn toàn corpus khớp bài gốc, nhưng phân bố từng nhãn trong từng split khác Table 7 của arXiv v2; cần dùng thống kê và hash của file thực tế khi công bố.

### 3.2. Kiểm tra trùng lặp và biến thể phục vụ GA

Không có văn bản hoặc nhãn bị thiếu. Train có 4 dòng trùng văn bản tính sau lần xuất hiện đầu; test có 1; validation không có. Có một chuỗi train–validation trùng nhau và một chuỗi trong train mang hai nhãn khác nhau. Không phát hiện trùng chính xác train–test hoặc validation–test. Kết quả giao nhau không đổi khi dùng NFC, chuyển chữ thường và gộp khoảng trắng để đối chiếu.

Gói giữ nguyên `data/` và thêm `data_variants/train_no_validation_overlap.csv`:

- Bỏ đúng `train_01030`, thuộc lớp Surprise, vì văn bản trùng validation.
- Train còn **5.547 mẫu**, Surprise còn 241; validation vẫn 686 và test vẫn 693.
- Không sửa văn bản, đổi nhãn hoặc tự xử lý các trường hợp mâu thuẫn.
- Script tạo biến thể chỉ đọc train và văn bản validation; không sử dụng nhãn validation để quyết định loại mẫu và không đọc test.

Đề xuất dùng biến thể này cho **tất cả phương pháp được so sánh trong nghiên cứu mới**. Nếu tái lập bảng theo bản phát hành nguyên gốc, chạy một bảng riêng với 5.548 train và công bố điểm trùng. Không so hai phương pháp trên hai phiên bản train khác nhau.

Đây chỉ là xử lý một điểm giao nhau đã biết; chưa có kiểm tra trùng ngữ nghĩa hoặc quan hệ cùng bài đăng. Không tuyên bố dữ liệu đã loại hết mọi khả năng rò rỉ. Có thể thêm thí nghiệm độ nhạy loại trùng trong train và bỏ các nhóm nhãn mâu thuẫn, với quy tắc chốt trước và số mẫu ghi rõ.

Độ dài train đo bằng tách khoảng trắng: trung vị 12, phân vị 95% là 33, dài nhất 164 đơn vị. Đây chưa phải số từ tiếng Việt hoặc số subword. Phải đo lại bằng tokenizer thực tế trước khi chọn độ dài tối đa.

### 3.3. Bộ mở rộng và điều kiện truy cập

**ViGoEmotions** có 20.664 bình luận, bài toán đa nhãn với 27 cảm xúc và Neutral, tức **28 đầu ra**. Nguồn này tái sử dụng 6.921 văn bản phân biệt từ UIT-VSMEC và gán nhãn lại. Vì vậy không coi nó là tập kiểm thử độc lập khi mô hình từng học UIT-VSMEC mà chưa loại giao nhau. [3]

Ngày 25/10/2026, trang [uitnlp/vigoemotions](https://huggingface.co/datasets/uitnlp/vigoemotions) vẫn yêu cầu đăng nhập và chấp nhận chia sẻ thông tin tài khoản để truy cập. Điều khoản chi tiết giới hạn nghiên cứu phi lợi nhuận và không cho phân phối lại; metadata MIT không thay thế phần điều khoản đó. **Gói bàn giao không chứa ViGoEmotions.** [4]

Khi được cấp quyền, dùng vector nhãn multi-hot, sigmoid và BCEWithLogitsLoss; ngưỡng chọn trên validation. Neutral phải có quy tắc loại trừ các nhãn khác theo hướng dẫn bộ dữ liệu. Không áp dụng argmax đơn nhãn của UIT-VSMEC cho trường hợp này.

ViTOED là hướng khác: nhận diện cảm xúc theo nguồn, đối tượng và biểu thức; không dùng thay trực tiếp bộ phân loại một nhãn cho cả câu. [5]

Với UIT-VSMEC, các workbook không kèm file giấy phép riêng; giữ nguồn và trích dẫn nghiên cứu, không tự gán quyền thương mại hoặc giấy phép MIT cho dữ liệu. Đây là ghi nhận về phần điều khoản đã đọc, không phải kết luận pháp lý về mọi cách sử dụng.

## 4. xLSTM có thể đóng vai trò gì?

xLSTM mở rộng LSTM bằng cổng hàm mũ được ổn định và hai dạng bộ nhớ: **sLSTM** dùng bộ nhớ vô hướng theo từng kênh cùng cơ chế trộn trạng thái; **mLSTM** dùng bộ nhớ ma trận. Các block được xếp chồng qua kết nối residual. Nghiên cứu gốc đánh giá cả các tác vụ tổng hợp, Long Range Arena và mô hình hóa ngôn ngữ; các kết quả đó không chứng minh sẵn hiệu quả cho cảm xúc tiếng Việt. [6]

Giả thuyết ứng dụng: một encoder tuần tự nhỏ có thể kết hợp dấu hiệu cảm xúc, phủ định và vị trí token trong câu; GA có thể tìm độ rộng, độ sâu và mức điều chuẩn phù hợp dữ liệu nhỏ. Tuy nhiên, corpus chủ yếu là câu ngắn, nên lợi thế nhớ chuỗi dài không phải bằng chứng xLSTM sẽ thắng. Đây là câu hỏi thực nghiệm.

Dùng `xLSTMBlockStack` của mã nguồn **NX-AI/xlstm** cho encoder phân loại. Nhánh “Models from the xLSTM NeurIPS Paper” phù hợp thiết kế này. `xLSTMLMModel` có head dự đoán token; cần head phân loại câu riêng. Kho mã cũng có xLSTM Large/7B, một lựa chọn khác với cấu hình encoder nhỏ đang đề xuất. [7]

Commit `main` quan sát ngày 25/10/2026: `ab22eadbd245f293dd8dec38ed29963d73758a12`. Đây là mốc nguồn để pin khi triển khai, **chưa được cài và kiểm thử huấn luyện trong gói này**.

### Kiến trúc khởi đầu được đề xuất

1. NFC và tokenizer SentencePiece unigram học **chỉ trên train đang sử dụng**; vocab khởi đầu 4.000. Giữ dấu, chữ hoa/thường, emoji và dấu câu. Ghi cấu hình normalization; ưu tiên NFC bên ngoài và normalization identity trong tokenizer để tránh biến đổi ngầm. Đo tỉ lệ unknown/truncation trên các tập. [8]
2. Token IDs `[B,T]` qua embedding thành `[B,T,d]`, với `d=128` khởi đầu. Embedding học cùng mạng, chưa dùng tiền huấn luyện.
3. Hai block mLSTM khởi đầu; mở rộng sLSTM/hybrid sau khi kiểm tra backend.
4. Right padding và **masked mean pooling**: lấy trung bình trên token hợp lệ; không để PAD tham gia mẫu số. Không dùng hidden tại vị trí cuối đã bị padding.
5. Dropout và linear ra logits `[B,7]`; dùng CrossEntropyLoss trên logits. Dự đoán bằng argmax; softmax chỉ cần khi xuất xác suất.

Vocab, tokenizer và preprocessing phải giống nhau khi so LSTM với xLSTM học từ đầu. Với mô hình tiền huấn luyện, sử dụng tokenizer và preprocessing được tác giả yêu cầu; PhoBERT cần đầu vào word-segmented. [9]

## 5. GA tối ưu gì và phối hợp với AdamW thế nào?

**GA tìm cấu hình; AdamW học trọng số mạng bên trong từng cấu hình.** Một cá thể là bộ siêu tham số, không phải toàn bộ hàng triệu trọng số. Với cấu hình `g`, train trên tập train rồi dùng validation để chấm điểm. Không dùng test để tính fitness.

Không gian giai đoạn đầu có sáu gene:

| Gene | Giá trị đề xuất | Ý nghĩa |
|---|---|---|
| `embedding_dim` | 64, 128, 192 | Độ rộng biểu diễn |
| `block_layout` | `m`, `mm`, `mmm` | Một, hai hoặc ba block mLSTM |
| `dropout` | 0,1; 0,3; 0,5 | Cùng một giá trị cho stack và head |
| `log10_learning_rate` | Số thực từ −4 đến −2,5 | Learning rate từ 0,0001 đến khoảng 0,00316 |
| `weight_decay` | 0; 0,0001; 0,001; 0,01 | Điều chuẩn trọng số |
| `class_weight_alpha` | 0; 0,5; 1 | Mức tăng trọng số cho lớp ít mẫu |

Ví dụ một cá thể, **chưa được chứng minh là tốt**:

```json
{
  "embedding_dim": 128,
  "block_layout": "mm",
  "dropout": 0.3,
  "log10_learning_rate": -3.3,
  "weight_decay": 0.0001,
  "class_weight_alpha": 0.5
}
```

Learning rate của ví dụ xấp xỉ `5.01e-4`. Với `N` mẫu train, `n_c` mẫu lớp c và 7 lớp, dùng `w_c = (N/(7*n_c))^alpha`, rồi chuẩn hóa để trung bình bảy trọng số bằng 1. Alpha bằng 0 tương ứng loss không cân bằng lớp; alpha bằng 0,5 giảm mức khuếch đại so với nghịch đảo tần suất đầy đủ. Chỉ tính `n_c` từ train. Không kết hợp thêm oversampling ở thí nghiệm chính vì sẽ khó tách ảnh hưởng của class weights.

Giữ cố định pooling, tokenizer, độ dài tối đa 128 token gồm EOS, 4 head, conv kernel 4, micro-batch 16, tích lũy gradient 2 bước, effective batch 32, gradient clip 1,0 và lịch learning rate hằng trong mỗi trial. Các tham số này là điểm khởi đầu, cần kiểm tra tài nguyên trước khi khóa protocol. Weight decay không áp dụng cho bias và tham số normalization.

Giai đoạn mở rộng có thể thay gene layout bằng `m`, `s`, `mm`, `ms`, `sm`, `ss`, `mmm`, `msm`, `sms`, `sss`. Khi đó số block bằng độ dài chuỗi; `slstm_at` là các vị trí ký tự `s`, đánh số từ 0. Phải chạy lại cả GA và random search trong cùng không gian mở rộng; không dùng ngân sách không gian cũ để tuyên bố phương pháp mới tốt hơn.

## 6. Fitness, toán tử và ngân sách cụ thể

### 6.1. Fitness

Chỉ số chính là **Macro-F1 trên validation**: tính F1 riêng cho đủ bảy lớp rồi lấy trung bình. Một lớp không có dự đoán đúng vẫn phải nằm trong phép tính; dùng `zero_division=0`. Accuracy và Weighted-F1 là chỉ số bổ sung. [10]

Trong vòng tìm kiếm nhanh, mỗi cấu hình train 5 epoch với cùng seed huấn luyện 42; fitness là Macro-F1 validation cao nhất trong năm epoch. Mọi thuật toán tìm kiếm dùng đúng quy tắc này. Không dùng training accuracy hoặc test F1 để xếp hạng.

Năm epoch có thể ưu ái mô hình học nhanh; đây là đánh giá chi phí thấp, không bảo đảm xếp hạng đúng sau hội tụ. Sau tìm kiếm, train lại top 3 cấu hình từ đầu với ba seed `[17,29,43]`, tối đa 30 epoch, patience 5 và chọn theo trung bình Macro-F1 validation. Báo cáo độ lệch chuẩn để thấy mức ổn định.

### 6.2. Một lần chạy GA

| Thành phần | Thiết lập khởi đầu |
|---|---|
| Quần thể | 8 cá thể |
| Thế hệ đầu | 8 cấu hình mới |
| Số vòng tạo thế hệ con | 4 |
| Giữ lại mỗi vòng | 2 elite |
| Con mới mỗi vòng | 6 |
| Tổng cấu hình được đánh giá | **8 + 4 × 6 = 32** |
| Chọn bố mẹ | Tournament 3 |
| Lai ghép | Uniform crossover, xác suất 0,8 cho mỗi cặp |
| Đột biến | Xác suất 1/6 cho mỗi gene |
| Gene rời rạc | Chọn một giá trị hợp lệ khác |
| Gene learning rate | Nhiễu Gaussian σ=0,3 trong log10, chặn biên, làm tròn 4 chữ số |

Hai elite giữ điểm đã có; không train lại để đủ 8 cá thể. Cache theo cấu hình đã chuẩn hóa, hash dữ liệu/tokenizer, seed, ngân sách train, code version và backend/precision. Cấu hình trùng không tốn một lần train mới; tiếp tục đề xuất cho đủ cấu hình mới. Lỗi OOM hoặc loss không hữu hạn được ghi nhận, có fitness âm vô cùng và **vẫn tính vào số trial đã thử**.

```text
Khởi tạo 8 cấu hình hợp lệ
Train từng cấu hình trên train; lấy fitness từ validation
Lặp 4 vòng:
    Giữ 2 cá thể có fitness tốt nhất
    Chọn bố mẹ → lai ghép → đột biến
    Chuẩn hóa cấu hình, kiểm tra cache và ràng buộc
    Đánh giá 6 cấu hình mới; tạo quần thể kế tiếp
Lấy 3 cấu hình tốt nhất đã quan sát
Train lại top 3 với 3 seed và ngân sách dài hơn
Khóa cấu hình thắng, train lại với 5 seed đã định
Đánh giá test và báo cáo tất cả 5 kết quả
```

Nếu không có bố mẹ khả thi, sinh lại ứng viên ngẫu nhiên trong miền đã kiểm tra. Không tự giảm batch hoặc đổi backend riêng cho một cá thể mà vẫn ghi cùng cấu hình. Cần sửa protocol chung rồi bắt đầu lại các phương pháp bị ảnh hưởng.

### 6.3. So sánh chi phí công bằng

Random search là đối chứng thiết yếu cho một thuật toán tìm siêu tham số; đây cũng là lý do nên bổ sung TPE khi đủ ngân sách. [11,12]

Cho GA/random/TPE cùng không gian, 32 trial, 5 epoch/trial, dữ liệu, seed train, quy tắc chọn checkpoint và bước train lại top 3. **32 trial không đồng nghĩa bằng GPU-hours**, vì độ rộng/độ sâu khác nhau. Ghi thời gian từng trial, đỉnh VRAM, số tham số và đường cong best validation F1 theo tổng GPU-seconds. Khi kết luận về hiệu quả tính toán, so ở các mốc thời gian chung hoặc dùng một giới hạn thời gian chung chốt trước.

Một lần tìm kiếm có tối đa 160 proxy-epoch; bước train lại top 3 có tối đa 270 epoch. Năm seed cuối thêm tối đa 150 epoch, tổng trên lý thuyết 580 epoch cho một pipeline; early stopping và tốc độ từng cấu hình làm thời gian thực tế khác nhau. Không quy đổi sang số giờ khi chưa đo trên GPU mục tiêu.

Tách hai nguồn ngẫu nhiên: seed tìm kiếm điều khiển quần thể; seed train điều khiển khởi tạo và minibatch. Pilot có thể chạy một search seed. Để đánh giá tính ổn định của **thuật toán GA**, lặp tìm kiếm với `[11,23,37]` cho mọi phương pháp; năm seed của một cấu hình chỉ đo độ biến động huấn luyện, không thay được các lần tìm kiếm độc lập.

## 7. Các thí nghiệm cần có để kết luận đúng

| Thí nghiệm | Câu hỏi được trả lời |
|---|---|
| TF-IDF + Linear SVM | Mốc đơn giản, rẻ trên dữ liệu này tốt đến đâu? |
| LSTM/BiLSTM học từ đầu | Kiến trúc truyền thống với cùng tokenizer/embedding hoạt động thế nào? |
| xLSTM cấu hình cố định | xLSTM ban đầu có lợi gì? |
| xLSTM + random search | Cải thiện đến từ việc thử nhiều cấu hình bao nhiêu? |
| xLSTM + GA | GA có tìm được cấu hình tốt hơn trong cùng điều kiện? |
| xLSTM + TPE | GA có cạnh tranh với một phương pháp HPO khác? |
| LSTM + GA | Lợi ích đến từ xLSTM hay GA cũng giúp LSTM tương tự? |
| PhoBERT-base và/hoặc ViSoBERT | Mức chất lượng khi dùng tiền huấn luyện tiếng Việt là bao nhiêu? |

So sánh kiến trúc cần kiểm soát tokenizer, embedding, dữ liệu, training và ngân sách tìm kiếm; ghi số tham số, thêm các cấu hình có số tham số gần nhau. So xLSTM học từ đầu với ViSoBERT fine-tune là so **hệ thống**, chưa tách được riêng ưu thế kiến trúc.

ViSoBERT công bố trên emotion recognition: Accuracy 68,10%, Weighted-F1 68,37%, Macro-F1 65,88% ở nghiên cứu năm 2023. Đây là số tác giả báo cáo, không phải kết quả chạy lại trong gói và không khẳng định là tốt nhất hiện nay. [13] Không đối chiếu trực tiếp con số test trong bài báo với baseline validation vừa chạy.

Các ablation ưu tiên: alpha bằng 0 so với alpha được chọn; layout chỉ mLSTM so với layout có sLSTM trong không gian mở rộng; cấu hình cố định so với random/GA; train nguyên gốc so với biến thể bỏ giao nhau. Thử mean/max/attention pooling là nhánh sau, với cùng ngân sách riêng; không thêm đồng thời nhiều thành phần rồi quy hết cải thiện cho GA.

Nếu encoder học từ đầu quá yếu, có thể thử **đóng băng cùng một ViSoBERT**, lấy chuỗi token features rồi so pooling+linear, LSTM và xLSTM. Không đưa duy nhất một vector câu vào xLSTM và coi đó là khai thác chuỗi. Cần báo cáo cả chi phí của encoder và head.

## 8. Giả thuyết và đóng góp nghiên cứu khả thi

- **H1:** GA cải thiện Macro-F1 so với random search ở ngân sách chung. Kiểm chứng qua các search seed độc lập, đường cong F1 theo thời gian và đánh giá cuối.
- **H2:** Tối ưu mức cân bằng lớp cải thiện F1 của lớp hiếm mà không làm giảm mạnh Macro-F1 tổng thể. So alpha bằng 0 và alpha được chọn với những thành phần còn lại giữ cố định.
- **H3:** Cấu trúc sLSTM/mLSTM phù hợp giúp đạt cân bằng chất lượng–chi phí tốt hơn một layout cố định. Cần ablation và kiểm soát số tham số.

Đây là giả thuyết có thể sai. Nếu GA không vượt random/TPE hoặc xLSTM không vượt LSTM ở điều kiện kiểm soát, cần báo cáo kết quả đó.

Tra cứu ngày 25/10/2026 chưa tìm được một công trình trùng khít “GA–xLSTM cho nhận diện cảm xúc văn bản tiếng Việt” trong những nguồn đã kiểm tra. **Điều này không đủ chứng minh chưa từng có nghiên cứu như vậy.** Đã có QGA kết hợp xLSTM cho bài toán danh mục đầu tư, vì thế không thể tuyên bố “lần đầu kết hợp GA và xLSTM” nói chung. [14]

Đóng góp nên nhấn vào protocol có thể tái lập, hiệu quả dưới giới hạn tài nguyên, ảnh hưởng mất cân bằng lớp và so sánh tìm kiếm công bằng. Nếu muốn mở rộng phương pháp, thử tối ưu đa mục tiêu gồm lỗi phân loại và chi phí, hoặc cơ chế phân bổ ngân sách nhiều mức; mỗi thay đổi phải có đối chứng với GA cơ bản.

## 9. Đánh giá cuối và kiểm soát độ tin cậy

Khóa mô hình, preprocessing, số epoch/patience, lựa chọn cấu hình, seed `[101,202,303,404,505]` và metric trước khi tạo dự đoán test. Báo cáo mean ± standard deviation của đủ năm seed; không chọn seed test tốt nhất. Nếu xem lỗi test rồi sửa mô hình, kết quả tiếp theo phải ghi rõ test đã tham gia phát triển.

Chỉ số cần xuất: Macro-F1 chính; Accuracy, Weighted-F1, precision/recall/F1 từng lớp, confusion matrix, thời gian train, số tham số, VRAM đỉnh và latency suy luận cùng batch/độ dài. Dùng script `evaluate.py` để ghép dự đoán theo `sample_id`, phát hiện ID thiếu/thừa/trùng và nhãn sai.

Với 693 mẫu test và các lớp chỉ vài chục mẫu, chênh lệch nhỏ có thể không ổn định. Có thể dùng paired bootstrap cho chênh lệch Macro-F1: lấy cùng các nhóm văn bản cho hai mô hình, giữ hai bản sao trong test cùng nhóm, tính chênh lệch theo từng seed rồi lấy trung bình. Khoảng tin cậy theo mẫu test này có điều kiện trên các mô hình đã train; báo cáo riêng độ lệch chuẩn giữa seed. Không coi cả hai là cùng một loại bất định.

Giữ test gốc 693 mẫu để đối chiếu. Nếu thêm kết quả trên 692 văn bản test phân biệt, ghi đó là phân tích bổ sung, không thay thế kết quả trên test phát hành.

## 10. Kết quả baseline đã thực chạy

Train: `train_no_validation_overlap`, **5.547 mẫu**. Validation: **686 mẫu**. TF-IDF gồm nhánh word n-gram 1–2 và character n-gram 3–5; `min_df=2`, tối đa 50.000 đặc trưng mỗi nhánh, sublinear TF. LinearSVC dùng `C=1`, `class_weight=balanced`, seed 42. Cấu hình được đặt trước khi đánh giá, chưa tìm siêu tham số.

| Chỉ số trên validation | Giá trị |
|---|---:|
| Accuracy | 57,58% |
| Macro-F1 | **53,19%** |
| Weighted-F1 | 57,55% |

Kết quả chính xác và dự đoán từng mẫu nằm trong `results/baseline_validation/`. Môi trường chạy: Python 3.12.14, pandas 2.2.3, scikit-learn 1.8.0; `run_metadata.json` ghi phiên bản và thời gian CPU. Đây là một lần chạy kiểm tra pipeline, không phải kết quả xLSTM, GA, hoặc test.

## 11. Triển khai trên GPU cá nhân

Với mức tài nguyên laptop 4 GB VRAM đã được nhắc trong bối cảnh trước, ưu tiên encoder nhỏ học từ đầu. Cần kiểm tra GPU thực tế trước khi dự toán thời gian; chưa có benchmark trên máy của người dùng.

1. Thử một minibatch forward/backward ở `d=64`, một block mLSTM, length 64; sau đó thử cấu hình rộng/sâu nhất của miền tìm kiếm với length 128 và micro-batch 16.
2. Đo loss hữu hạn, gradient, đỉnh VRAM và thời gian/epoch. Chọn miền khả thi trước khi chạy tất cả thuật toán tìm kiếm.
3. Khóa backend/precision chung. FP32 làm mốc kiểm tra số học; chỉ dùng mixed precision khi backend hỗ trợ và loss ổn định. Gradient accumulation không khắc phục được OOM bên trong một micro-batch.
4. sLSTM CUDA chính thức yêu cầu compute capability từ 8.0 và môi trường PyTorch/CUDA phù hợp. Ghi capability thực tế và thử kernel; không suy ra chạy được chỉ từ tên GPU. [7]
5. Nếu sLSTM không chạy ổn định, hoàn thành nghiên cứu mLSTM-only và ghi phạm vi đúng; mở rộng hybrid sau. Chi phí biên dịch/setup và chi phí train phải được ghi riêng, áp dụng cùng quy ước cho các phương pháp.

Gói chưa cài PyTorch/xLSTM, chưa tải trọng số tiền huấn luyện và chưa chạy tìm kiếm GPU. JSON trong `configs/` là đặc tả để triển khai, không phải trainer có thể gọi trực tiếp.

## 12. Cách dùng gói và lộ trình

Từ thư mục `UIT_VSMEC` sau khi giải nén:

```bash
python -m pip install -r requirements.txt
python scripts/prepare_training_variant.py
python scripts/baseline_tfidf.py
```

Đọc dữ liệu:

```python
import pandas as pd

train = pd.read_csv("data_variants/train_no_validation_overlap.csv", keep_default_na=False)
validation = pd.read_csv("data/validation.csv", keep_default_na=False)
# text là đầu vào; label_id là nhãn số. Không đưa ID hoặc source_index vào mô hình.
```

Khi đã có dự đoán, lưu hai cột `sample_id,pred_label`, trong đó `pred_label` là tên nhãn:

```bash
python scripts/evaluate.py --gold data/validation.csv --pred predictions_validation.csv --output results/validation_metrics.json
```

Chỉ sau khi khóa protocol mới đánh giá test:

```bash
python scripts/evaluate.py --gold data/test.csv --pred predictions_test.csv --output results/test_metrics.json
```

Lộ trình thực hiện: (1) khóa data/tokenizer và tái lập baseline; (2) triển khai LSTM/xLSTM nhỏ và đo tài nguyên; (3) chạy random search và GA với cùng protocol; (4) train lại, đánh giá cuối, phân tích ablation và viết báo cáo. Một bộ kết quả NCKH cần có bảng so sánh, log của mọi trial, hash dữ liệu/cấu hình/code, seed, môi trường và dự đoán cuối; không chỉ file model tốt nhất.

## Tài liệu tham khảo và nguồn

1. Ho et al. **Emotion Recognition for Vietnamese Social Media Text**, PACLING 2019/ấn bản 2020. [arXiv](https://arxiv.org/abs/1911.09339), [DOI](https://doi.org/10.1007/978-981-15-6168-9_27).
2. [Thư mục UIT-VSMEC](https://drive.google.com/drive/folders/1HooABJyrddVGzll7fgkJ6VzkG_XuWfRu); [trang UIT NLP dùng ở lần tải trước](https://nlp.uit.edu.vn/datasets); [danh mục của tác giả](https://github.com/kietnv/VietnameseDatasets). URL từng workbook và hash nằm trong `sources.json`.
3. Hung et al. **ViGoEmotions**, EACL 2026. [ACL](https://aclanthology.org/2026.eacl-long.129/), [toàn văn arXiv v2](https://arxiv.org/html/2602.08371v2).
4. [ViGoEmotions dataset card và điều kiện truy cập](https://huggingface.co/datasets/uitnlp/vigoemotions); [mã nguồn tác giả](https://github.com/ricardo-tran/ViGoEmotions).
5. Vo et al. **ViTOED**, 2026. [arXiv](https://arxiv.org/abs/2608.12776).
6. Beck et al. **xLSTM: Extended Long Short-Term Memory**, NeurIPS 2024. [arXiv](https://arxiv.org/abs/2405.04517).
7. [NX-AI/xlstm](https://github.com/NX-AI/xlstm); [commit quan sát](https://github.com/NX-AI/xlstm/commit/ab22eadbd245f293dd8dec38ed29963d73758a12).
8. [SentencePiece chính thức](https://github.com/google/sentencepiece).
9. [PhoBERT chính thức và yêu cầu word segmentation](https://github.com/VinAIResearch/PhoBERT).
10. [scikit-learn: F1-score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html); [LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html); [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).
11. Bergstra & Bengio. **Random Search for Hyper-Parameter Optimization**, JMLR 2012. [Bài báo](https://jmlr.org/papers/v13/bergstra12a.html).
12. Bergstra et al. **Algorithms for Hyper-Parameter Optimization**, NeurIPS 2011. [Bài báo](https://proceedings.neurips.cc/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html).
13. Nguyen et al. **ViSoBERT**, EMNLP 2023. [ACL](https://aclanthology.org/2023.emnlp-main.315/), [Table 4](https://arxiv.org/html/2310.11166v1).
14. Li, Song & Feng. **QGA-optimised BL_xLSTM_MLP model for portfolio**, IJCSE 2026. [Trang nhà xuất bản](https://www.inderscience.com/info/inarticle.php?artid=152472), DOI `10.1504/IJCSE.2026.152472`. Đã đọc abstract công khai, chưa đọc toàn văn bị giới hạn truy cập.

Các quyết định cấu hình, fitness, ngân sách, seed và lộ trình là đề xuất trong tài liệu này, không phải thiết lập đã được các bài báo trên xác nhận là tối ưu.
