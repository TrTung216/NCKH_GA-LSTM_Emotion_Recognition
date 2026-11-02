"""Build a Vietnamese research report from saved, verified experimental results."""
import json
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.opc.constants import RELATIONSHIP_TYPE as RT
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports'
ASSETS = OUT / 'assets'
FIXED = ROOT / 'results/lstm_xlstm_comparison'
GA = ROOT / 'results/ga_lstm_xlstm_comparison'
DEST = OUT / 'Bao_cao_thuc_nghiem_LSTM_xLSTM_GA_UIT_VSMEC.docx'
ORDER = ['LSTM', 'GA+LSTM', 'xLSTM', 'GA+xLSTM']
LABELS = ['Anger', 'Disgust', 'Enjoyment', 'Fear', 'Other', 'Sadness', 'Surprise']


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def num(value, digits=2):
    return f'{value:.{digits}f}'.replace('.', ',')


def pct(value):
    return num(value * 100)


def shading(cell, color):
    element = OxmlElement('w:shd'); element.set(qn('w:fill'), color)
    cell._tc.get_or_add_tcPr().append(element)


def add_link(paragraph, label, url):
    relation = paragraph.part.relate_to(url, RT.HYPERLINK, is_external=True)
    link = OxmlElement('w:hyperlink'); link.set(qn('r:id'), relation)
    run = OxmlElement('w:r'); props = OxmlElement('w:rPr')
    color = OxmlElement('w:color'); color.set(qn('w:val'), '176A88'); props.append(color)
    run.append(props); text = OxmlElement('w:t'); text.text = label; run.append(text)
    link.append(run); paragraph._p.append(link)


def build():
    OUT.mkdir(exist_ok=True); ASSETS.mkdir(exist_ok=True)
    evidence = read(OUT / 'analysis_evidence.json')
    raw = pd.read_csv(GA / 'comparison_runs.csv')
    raw['method'] = raw.method.replace({'XLSTM': 'xLSTM', 'GA+XLSTM': 'GA+xLSTM'})
    means = raw.groupby('method').agg({'validation_macro_f1': ['mean','std'], 'test_macro_f1': ['mean','std'],
        'test_accuracy':['mean','std'], 'test_weighted_f1':['mean','std']})
    selected = read(GA / 'selected_configs.json')['selected']
    costs = pd.read_csv(GA / 'costs.csv').set_index('model')
    perclass = pd.read_csv(GA / 'per_class_test.csv')
    perclass['method'] = perclass.method.replace({'XLSTM':'xLSTM','GA+XLSTM':'GA+xLSTM'})
    perclass = perclass.pivot(index='label', columns='method', values='f1_mean').loc[LABELS]
    train = pd.read_csv(ROOT / 'data_variants/train_no_validation_overlap.csv')
    distribution = pd.read_csv(ROOT / 'audit/label_distribution.csv').set_index('label')
    distribution['train'] = train.label.value_counts()
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    sec.top_margin, sec.bottom_margin = Cm(1.8), Cm(1.7)
    sec.left_margin, sec.right_margin = Cm(2.2), Cm(1.8)
    sec.header_distance, sec.footer_distance = Cm(.7), Cm(.7)
    sec.different_first_page_header_footer = True
    normal = doc.styles['Normal']
    normal.font.name = 'Times New Roman'; normal.font.size = Pt(11)
    normal.paragraph_format.line_spacing = 1.12
    normal.paragraph_format.space_after = Pt(5)
    for name, size in [('Title',27), ('Subtitle',15), ('Heading 1',16), ('Heading 2',12), ('Heading 3',11)]:
        style = doc.styles[name]; style.font.name = 'Arial'; style.font.size = Pt(size)
        style.font.color.rgb = RGBColor.from_string('164863')
        style.paragraph_format.space_before = Pt(10)
        style.paragraph_format.space_after = Pt(7)
    doc.styles['Caption'].font.name = 'Times New Roman'
    doc.styles['Caption'].font.size = Pt(9)
    doc.styles['Caption'].font.color.rgb = RGBColor.from_string('526370')
    lang = OxmlElement('w:lang'); lang.set(qn('w:val'), 'vi-VN')
    normal.element.get_or_add_rPr().append(lang)
    header = sec.header.paragraphs[0]
    header.text = 'UIT-VSMEC  |  BÁO CÁO THỰC NGHIỆM LSTM, xLSTM VÀ GA'
    header.style = doc.styles['Caption']
    footer = sec.footer.paragraphs[0]; footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    footer.add_run('28/10/2026  •  Trang ')
    field = OxmlElement('w:fldSimple'); field.set(qn('w:instr'), 'PAGE'); footer._p.append(field)
    for run in footer.runs: run.font.size = Pt(9)
    doc.core_properties.title = 'Thực nghiệm LSTM, xLSTM, GA+LSTM và GA+xLSTM trên UIT-VSMEC'
    doc.core_properties.subject = 'Phương pháp, kết quả, phân tích khác biệt và hướng cải thiện'
    doc.core_properties.author = 'Báo cáo tổng hợp từ thực nghiệm UIT-VSMEC'
    doc.core_properties.keywords = 'UIT-VSMEC, LSTM, xLSTM, mLSTM, GA, cảm xúc tiếng Việt'

    def p(text='', bold=False, style=None):
        para = doc.add_paragraph(style=style)
        para.add_run(text).bold = bold
        return para

    def h(text, level=2): doc.add_heading(text, level)

    def page(title):
        doc.add_page_break(); h(title, 1)

    def note(text):
        para = p(text)
        para.paragraph_format.left_indent = Cm(.25)
        para.paragraph_format.right_indent = Cm(.15)
        props = para._p.get_or_add_pPr()
        shade = OxmlElement('w:shd'); shade.set(qn('w:fill'), 'EAF3F7'); props.append(shade)

    def table(headers, rows, widths=None, size=9.4):
        t = doc.add_table(rows=1, cols=len(headers))
        t.alignment = WD_TABLE_ALIGNMENT.CENTER; t.autofit = False
        widths = widths or [17 / len(headers)] * len(headers)
        for col, width in zip(t.columns, widths): col.width = Cm(width)
        for cell, title, width in zip(t.rows[0].cells, headers, widths):
            cell.text = str(title); cell.width = Cm(width); shading(cell, '164863')
        repeat = OxmlElement('w:tblHeader'); t.rows[0]._tr.get_or_add_trPr().append(repeat)
        for i, row in enumerate(rows):
            cells = t.add_row().cells
            for cell, value, width in zip(cells, row, widths):
                cell.text = str(value); cell.width = Cm(width)
                if i % 2 == 0: shading(cell, 'F0F5F7')
        for i, row in enumerate(t.rows):
            no_split = OxmlElement('w:cantSplit'); row._tr.get_or_add_trPr().append(no_split)
            for cell in row.cells:
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                for para in cell.paragraphs:
                    para.paragraph_format.space_after = Pt(3)
                    para.paragraph_format.space_before = Pt(3)
                    para.paragraph_format.line_spacing = 1.02
                    for run in para.runs:
                        run.font.name = 'Times New Roman'; run.font.size = Pt(size)
                        if i == 0: run.bold = True; run.font.color.rgb = RGBColor(255,255,255)
        p().paragraph_format.space_after = Pt(0)
        return t

    def picture(path, caption, width=16.5):
        para = doc.add_paragraph(); para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.keep_with_next = True
        para.add_run().add_picture(str(path), width=Cm(width))
        p(caption, style='Caption')

    def score(method, metric):
        return pct(means.loc[method,(metric,'mean')]) + ' ± ' + pct(means.loc[method,(metric,'std')])

    # 1. Cover and executive findings.
    p('BÁO CÁO NGHIÊN CỨU THỰC NGHIỆM', style='Subtitle')
    p('PHÂN LOẠI CẢM XÚC\nVĂN BẢN TIẾNG VIỆT', style='Title')
    p('LSTM • xLSTM • GA+LSTM • GA+xLSTM\nBộ dữ liệu UIT-VSMEC', style='Subtitle')
    p('Ngày thực nghiệm và tổng hợp: 28/10/2026')
    p('Báo cáo sử dụng kết quả huấn luyện thực tế, dự đoán theo ID và nhật ký đã lưu trong dự án. Phần phân tích bổ sung chỉ tính thống kê từ kết quả có sẵn; không huấn luyện lại và không điều chỉnh mô hình theo test.')
    h('Các phát hiện chính')
    table(['So sánh', 'Macro-F1 test', 'Diễn giải trong phạm vi thực nghiệm'], [
        ['LSTM → xLSTM', '44,15% → 46,29%', 'xLSTM cấu hình cố định cao hơn 2,14 điểm %, nhưng tốn thời gian và bộ nhớ hơn.'],
        ['LSTM → GA+LSTM', '44,15% → 48,32%', 'Tăng 4,17 điểm %; GA+LSTM đứng đầu bốn cấu hình được đánh giá.'],
        ['xLSTM → GA+xLSTM', '46,29% → 46,07%', 'Không tăng Macro-F1 trung bình dù Accuracy tăng; lớp Anger suy giảm rõ.']
    ], [3.6,3.4,10])
    note('Kết luận trọng tâm: kết quả phụ thuộc cả kiến trúc, siêu tham số và quy trình chọn mô hình. Kiến trúc mới hơn hoặc thêm GA không tự bảo đảm điểm tốt hơn trên một bộ dữ liệu nhỏ.')
    p('Phạm vi: bảy lớp cảm xúc, embedding học từ đầu, xLSTM chỉ dùng mLSTM; một lần tìm kiếm GA cho mỗi kiến trúc và ba seed huấn luyện cuối. Kết quả là nghiên cứu thăm dò, chưa phải kết luận phổ quát hoặc xác nhận trên một test mới.')

    # 2. Questions and interpretation.
    page('1. Mục tiêu và cách đọc báo cáo')
    p('Mục tiêu là đo sự khác biệt giữa LSTM và xLSTM trong cùng bài toán phân loại một nhãn trong bảy lớp, sau đó kiểm tra tác động của tìm siêu tham số bằng thuật toán di truyền (GA). Báo cáo trả lời cả câu hỏi về độ chính xác lẫn chi phí tính toán.')
    table(['Câu hỏi', 'Đối chứng và bằng chứng'], [
        ['xLSTM có tốt hơn LSTM khi chưa tìm siêu tham số?', 'Giữ dữ liệu, tokenizer, chiều embedding, số tầng và lịch train chung; so sánh ba seed cùng giá trị.'],
        ['GA giúp từng kiến trúc ở mức nào?', 'So cấu hình được GA chọn với cấu hình cố định của cùng kiến trúc; giữ tokenizer và tiêu chí chọn checkpoint.'],
        ['GA+LSTM hay GA+xLSTM phù hợp hơn?', 'So điểm tổng thể, từng lớp, độ dao động giữa seed, số tham số, thời gian và bộ nhớ.'],
        ['Vì sao có khác biệt?', 'Phân rã Macro-F1 theo lớp, kiểm tra confusion matrix, độ dài câu và đường học; tách bằng chứng mô tả khỏi giả thuyết nhân quả.']
    ], [6,11])
    h('1.1. Ba mức độ kết luận')
    p('Quan sát trực tiếp: một con số được tính lại từ metrics, predictions hoặc history. Ví dụ F1 Anger của GA+xLSTM thấp hơn xLSTM 11,00 điểm %.')
    p('Giải thích có cơ sở: cách cấu thành chỉ số hoặc hành vi được nhật ký hỗ trợ. Ví dụ Anger giảm làm kéo Macro-F1 xuống vì mỗi lớp có trọng số 1/7 trong trung bình.')
    p('Giả thuyết cần kiểm chứng: mô hình có thể quá nhạy với learning rate, thiếu biểu diễn ngữ nghĩa hoặc cần dữ liệu lớn hơn. Chưa thể xác nhận các nguyên nhân này khi nhiều siêu tham số cùng thay đổi và chưa có ablation (thay đổi có kiểm soát từng yếu tố).')
    h('1.2. Quy ước báo cáo')
    p('Các điểm được nhân 100 và ghi bằng %. Chênh lệch giữa hai điểm là “điểm phần trăm”, không phải phần trăm tăng tương đối. Dấu ± là độ lệch chuẩn mẫu giữa ba seed (ddof=1), không phải khoảng tin cậy và không phải kết quả kiểm định ý nghĩa thống kê.')
    p('Macro-F1 là tiêu chí chính để chọn cấu hình/checkpoint. Accuracy và Weighted-F1 là chỉ số bổ sung. Mọi giá trị từng lớp được trung bình từ ba lần đánh giá riêng; không gộp các dự đoán thành một ensemble.')
    note('Cấu trúc: mục 2–4 mô tả phương pháp; mục 5 so sánh LSTM/xLSTM; mục 6–7 trình bày GA và nguyên nhân chênh lệch; mục 8 nêu giới hạn; mục 9–10 kết luận và đề xuất; cuối báo cáo là tài liệu tham khảo và phụ lục kết quả từng seed.')

    # 3. Dataset and metric.
    page('2. Dữ liệu, tiền xử lý và chỉ số đánh giá')
    p('UIT-VSMEC là corpus cảm xúc văn bản mạng xã hội tiếng Việt do Ho và cộng sự giới thiệu [1]. Báo cáo dùng bản workbook nor_811 đã có trong dự án, giữ nguyên validation/test; không đối chiếu trực tiếp điểm với bài gốc vì quy trình và biến thể train có thể khác.')
    table(['Nhãn', 'Train thực dùng', 'Validation', 'Test'], [[name, int(distribution.loc[name,'train']), int(distribution.loc[name,'validation']), int(distribution.loc[name,'test'])] for name in LABELS] + [['Tổng',5547,686,693]], [5.2,4,3.9,3.9])
    p('Train gốc có 5.548 mẫu. Bỏ đúng train_01030 do trùng văn bản với validation, còn 5.547 mẫu; số mẫu Surprise trong train giảm từ 242 xuống 241. Mẫu trùng trong train và trường hợp nhãn mâu thuẫn được ghi nhận nhưng không tự đổi nhãn. Other không được tự đổi thành Neutral.')
    p('Tokenization: NFC → SentencePiece unigram chỉ học trên train → EOS → cắt tối đa 128 token → PAD bên phải. Yêu cầu tối đa 4.000 token, vocabulary thực tế 2.845. Giữ dấu tiếng Việt, chữ hoa, dấu câu và emoji; không xóa stopword hay từ phủ định.')
    table(['Tập', 'Trung vị token', 'Phân vị 95%', 'Dài nhất', 'Bị cắt >128'], [[split, num(v['median_tokens_including_eos'],0), num(v['p95_tokens_including_eos'],1), v['maximum'],v['over_128']] for split,v in evidence['lengths'].items()], [4,3.3,3.3,3.2,3.2])
    p('Độ dài tính bằng tokenizer thực tế và có EOS. Phần lớn câu ngắn: trung vị train là 15 token. Không có câu test bị cắt; vì vậy không có bằng chứng việc cắt chuỗi trực tiếp gây lỗi test trong thí nghiệm này.')
    h('2.1. Chỉ số')
    p('F1_c = 2 × Precision_c × Recall_c / (Precision_c + Recall_c). Macro-F1 = (1/7) × ΣF1_c. Weighted-F1 = Σ(n_c/N) × F1_c. Accuracy = số mẫu dự đoán đúng/N; mẫu số n_c là support thật của lớp c trên tập đang đánh giá.')
    p('Enjoyment có 193 mẫu test, Anger chỉ có 40. Do đó tăng đúng ở lớp đông mẫu có thể làm Accuracy tăng trong khi F1 một lớp ít mẫu giảm kéo Macro-F1 xuống. Cả hai chỉ số đều cần được đọc cùng báo cáo từng lớp.')

    # 4. Model and common training.
    page('3. Kiến trúc và thiết lập huấn luyện chung')
    p('Luồng xử lý: token IDs → embedding học từ đầu → encoder một chiều → trung bình có mask trên các token hợp lệ → dropout → lớp tuyến tính 7 đầu ra. Huấn luyện bằng cross-entropy trên logits; suy luận lấy nhãn có logit lớn nhất.')
    table(['Thành phần', 'LSTM', 'xLSTM trong thực nghiệm'], [
        ['Bộ nhớ', 'Trạng thái ô nhớ dạng vector, cập nhật qua các cổng; dùng torch.nn.LSTM [2].', 'Hai block mLSTM trong cấu hình cố định: bộ nhớ ma trận, cổng hàm mũ được ổn định, kết nối residual [3].'],
        ['Chuẩn hóa', 'LayerNorm sau encoder.', 'Chuẩn hóa tích hợp trong stack chính thức.'],
        ['Nguồn triển khai', 'PyTorch 2.5.1+cu121.', 'NX-AI/xlstm, commit ab22ead…; backend mLSTM song song bằng PyTorch [4].'],
        ['Phạm vi', 'Một chiều; chưa thử BiLSTM.', 'mLSTM-only; chưa thử sLSTM, tổ hợp m/s hoặc xLSTM 7B.']
    ], [3.1,6.7,7.2])
    p('Cổng và bộ nhớ của xLSTM cung cấp cơ chế biểu diễn khác LSTM, nhưng bài báo kiến trúc không chứng minh sẵn lợi thế trên UIT-VSMEC. Khác biệt quan sát ở đây thuộc về toàn bộ cấu hình encoder, chuẩn hóa và dropout, không riêng bộ nhớ ma trận [3].')
    table(['Thiết lập cố định', 'Giá trị'], [
        ['Embedding / tầng / dropout', '64 chiều / 2 tầng hoặc block / 0,3'],
        ['Optimizer / learning rate / weight decay', 'AdamW / 0,001 / 0,0001; loại bias và tham số chuẩn hóa khỏi weight decay'],
        ['Batch / precision / clipping', '32 / FP32 / gradient norm tối đa 1,0'],
        ['Trọng số lớp', 'w_c = (N/(7n_c))^α; α=0,5, rồi chuẩn hóa trung bình trọng số bằng 1; n_c chỉ lấy từ train'],
        ['Epoch và chọn checkpoint', 'Tối đa 20; patience 5; chọn Macro-F1 validation cao nhất, hòa giữ epoch sớm hơn'],
        ['Seed và thiết bị', '42, 123, 2026; NVIDIA GeForce RTX 3050 Ti Laptop GPU 4 GB']
    ], [6,11])
    p('Các seed có cùng khởi tạo embedding giữa hai kiến trúc khi cùng kích thước, cùng dữ liệu và thứ tự lấy batch theo seed. Có kiểm tra padding/batch invariance và gradient hữu hạn. Mã nguồn mLSTM không bị sửa phép tính; chỉ trì hoãn import sLSTM CUDA để tránh yêu cầu bộ biên dịch không cần dùng.')
    note('Giới hạn tái lập: CUDA cumsum trong mLSTM của môi trường hiện tại không có kernel deterministic; dùng warn_only. Cố định seed giúp tái lập quy trình nhưng không cam kết kết quả giống từng bit.')

    # 5. GA protocol.
    page('4. Thiết kế thực nghiệm GA')
    p('GA tối ưu siêu tham số; trọng số mạng vẫn được AdamW học. Mỗi cá thể là một bộ sáu gene. Cả hai kiến trúc dùng cùng không gian, quần thể ban đầu và ngân sách số trial; từ thế hệ tiếp theo, quần thể thay đổi theo fitness của từng kiến trúc.')
    spec = read(GA / 'locked_protocol.json')['spec']
    table(['Gene', 'Giá trị được phép'], [[k, ', '.join(str(v) for v in vals)] for k,vals in spec['search_space'].items()], [6,11])
    p('Không gian gồm 2 × 2 × 3 × 3 × 3 × 3 = 324 cấu hình. Mỗi kiến trúc chỉ thử 18 cấu hình duy nhất (khoảng 5,56% không gian), nên kết quả là nghiệm tốt trong phạm vi đã tìm, không phải bảo đảm tối ưu toàn cục.')
    table(['Giai đoạn', 'Quy tắc và ngân sách'], [
        ['Tìm kiếm GA', 'Quần thể 6; 3 thế hệ sinh thêm; giữ 2 elite; 6 + 3×4 = 18 trial. Mỗi trial 5 epoch; không dừng sớm trong proxy.'],
        ['Chọn/lai/đột biến', 'Tournament 3; uniform crossover xác suất 0,8; mỗi gene đột biến xác suất 1/6 sang giá trị khác. Đề xuất trùng bị loại.'],
        ['Fitness và seed', 'Macro-F1 validation tốt nhất trong 5 epoch; seed GA 71, seed train proxy 42. Trial lỗi vẫn tính ngân sách.'],
        ['Chạy lại ứng viên', 'Top 2 được train từ đầu với seed 17, 29; tối đa 20 epoch, patience 5; chọn theo trung bình validation Macro-F1.'],
        ['Đánh giá cuối', 'Khóa cấu hình của cả hai kiến trúc; train từ đầu trên 42, 123, 2026. Chỉ đọc test sau khi cả 6 run cuối hoàn tất.']
    ], [4.3,12.7])
    p('Khối lượng thực chạy: 36 trial proxy + 8 run chọn cấu hình + 6 run cuối = 50 lần huấn luyện. Không có trial lỗi. Hai baseline cố định trước đó có thêm 6 run, tổng cộng 56 run huấn luyện cho bốn phương pháp.')
    p('Các bộ dự đoán đã được chấm lại bằng bộ đánh giá độc lập theo sample_id: 12 bộ của baseline và 56 bộ của thí nghiệm GA. Số bộ dự đoán khác số run vì một run cuối có cả validation và test.')
    note('Test đã được xem ở giai đoạn baseline. GA không đọc nhãn test trong vòng tìm kiếm, nhưng nghiên cứu mở rộng này không có test hoàn toàn mới; kết luận cần được xem là thăm dò.')

    # Remaining sections are appended below.
    add_results(doc, p, h, page, note, table, picture, score, raw, means, perclass, evidence, selected, costs)
    doc.save(DEST)
    source_paths = [FIXED/'summary_runs.csv', GA/'comparison_mean_std.csv', GA/'per_class_test.csv',
                    GA/'costs.csv', GA/'selected_configs.json', OUT/'analysis_evidence.json']
    manifest = {'docx': str(DEST.relative_to(ROOT)), 'sha256': hashlib.sha256(DEST.read_bytes()).hexdigest(),
        'source_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
        'tables': len(doc.tables), 'images': len(doc.inline_shapes), 'paragraphs': len(doc.paragraphs)}
    (OUT/'docx_build_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def add_results(doc, p, h, page, note, table, picture, score, raw, means, perclass, evidence, selected, costs):
    # 6. Fixed comparison.
    page('5. LSTM và xLSTM: kết quả cấu hình cố định')
    table(['Mô hình', 'Val Macro-F1', 'Test Accuracy', 'Test Macro-F1', 'Test Weighted-F1'],
          [[m,score(m,'validation_macro_f1'),score(m,'test_accuracy'),score(m,'test_macro_f1'),score(m,'test_weighted_f1')] for m in ['LSTM','xLSTM']], [2.4,3.6,3.6,3.7,3.7])
    p('xLSTM cao hơn LSTM 2,14 điểm % Macro-F1 test và 1,68 điểm % Accuracy trung bình. Tuy nhiên, kết quả theo cặp seed không đồng nhất: xLSTM thấp hơn ở seed 42, cao hơn ở seed 123 và 2026. Chênh lệch lớn nhất nằm ở seed 123, nên không nên diễn giải trung bình là lợi thế chắc chắn trong mọi lần train.')
    rows = []
    for seed in [42,123,2026]:
        a = raw[(raw.method=='LSTM') & (raw.seed==seed)].iloc[0]
        b = raw[(raw.method=='xLSTM') & (raw.seed==seed)].iloc[0]
        rows.append([seed,pct(a.test_macro_f1),pct(b.test_macro_f1),num((b.test_macro_f1-a.test_macro_f1)*100)])
    table(['Seed','LSTM F1 test','xLSTM F1 test','xLSTM − LSTM (điểm %)'], rows, [2.5,4,4,6.5])
    picture(FIXED/'learning_curves.png','Hình 1. Macro-F1 validation qua các epoch của ba seed. Đường học dùng để chọn checkpoint; không dùng test.',16.5)
    table(['Mô hình','Tham số','Train TB (giây)','GPU peak TB (MiB)'], [[m,int(raw[raw.method==m].parameters.iloc[0]),num(raw[raw.method==m].training_seconds_including_validation_and_checkpoints.mean(),1),num(raw[raw.method==m].peak_cuda_allocated_mb.mean(),1)] for m in ['LSTM','xLSTM']], [3.5,4,4.7,4.8])
    p('xLSTM có ít tham số hơn một chút (242.903 so với 249.223), nhưng thời gian train trung bình cao hơn 3,78 lần và bộ nhớ tensor GPU cao hơn khoảng 2,23 lần. Thời gian đã gồm validation và ghi checkpoint; đây không phải tốc độ inference hay tổng VRAM tiến trình.')

    # 7. Fixed causes.
    page('5.1. Điểm mạnh, điểm yếu và giải thích khác biệt')
    table(['Nhãn','LSTM F1','xLSTM F1','Chênh lệch (điểm %)'], [[c,pct(perclass.loc[c,'LSTM']),pct(perclass.loc[c,'xLSTM']),num((perclass.loc[c,'xLSTM']-perclass.loc[c,'LSTM'])*100)] for c in LABELS], [4.4,3.6,3.6,5.4])
    h('Chất lượng dự đoán')
    p('Điểm mạnh quan sát của xLSTM là Anger (+7,43 điểm %), Fear (+5,80) và Enjoyment (+4,44). LSTM lại tốt hơn ở Other (+4,15 điểm % so với xLSTM) và Sadness (+1,97). Do đó lợi thế tổng thể không đồng nghĩa thắng trên mọi cảm xúc.')
    p('Ma trận nhầm lẫn hỗ trợ một giải thích cụ thể: ở Anger, recall tăng từ 32,50% với LSTM lên 40,83% với xLSTM; tỉ lệ Anger bị dự đoán thành Disgust giảm từ 39,17% xuống 31,67% (trung bình ba seed). xLSTM nhận ra thêm các mẫu Anger trong tập test này, nhưng chưa biết đặc trưng ngôn ngữ nào tạo ra khác biệt.')
    h('Hiệu quả tính toán')
    p('Kiểm tra mã thực thi cho thấy backend mLSTM tạo các tensor có hai chiều thời gian S×S, gồm ma trận cổng và tích query–key. LSTM dùng toán tử thư viện PyTorch; hai mô hình không sử dụng cùng một loại kernel. Vì vậy số tham số nhỏ hơn không kéo theo bộ nhớ trung gian hoặc thời gian nhỏ hơn [4].')
    p('Đây là giải thích ở mức triển khai, phù hợp với số đo nhưng chưa phải phân tích profiler. Chưa đo thời gian từng kernel, chưa benchmark inference, và các lần đo không khóa xung nhịp GPU hay tải nền. Không suy ra mọi triển khai xLSTM đều chậm hơn LSTM.')
    h('Giả thuyết kiến trúc cần kiểm chứng')
    p('Bộ nhớ ma trận, convolution nhân quả và residual của mLSTM có thể tạo cách kết hợp tín hiệu khác, phù hợp một số lớp. Tuy nhiên, chưa có ablation để tách riêng các thành phần. Bộ dữ liệu chủ yếu câu ngắn và mô hình học embedding từ đầu; chưa có bằng chứng lợi thế quan sát là do ghi nhớ phụ thuộc xa [3].')
    note('Kết luận phần này: xLSTM cấu hình cố định nhỉnh hơn về Macro-F1 trung bình và ít dao động hơn trong ba seed đã chạy; LSTM tiết kiệm tài nguyên hơn. Đây là so sánh hai cấu hình cụ thể, không phải xếp hạng tuyệt đối hai họ kiến trúc.')

    # 8. GA stage results.
    page('6. GA: kết quả từng giai đoạn tìm kiếm')
    generation_rows = []
    states = {m:read(GA/'search'/m/'search_state.json') for m in ['lstm','xlstm']}
    for generation in range(4):
        values = []
        for kind in ['lstm','xlstm']:
            values.append(pct(max(r['fitness'] for r in states[kind]['trials'] if r['generation']<=generation)))
        generation_rows.append([generation,6+generation*4,*values])
    table(['Thế hệ','Trial tích lũy/mô hình','LSTM: best proxy F1','xLSTM: best proxy F1'],generation_rows,[2.5,4.4,5,5.1])
    p('Fitness trong bảng là điểm validation tốt nhất của các trial 5 epoch đã hoàn tất, không phải điểm test. Cấu hình baseline được đưa vào quần thể ban đầu để có mốc so sánh ở cùng ngân sách ngắn. Tất cả 36 trial đều thành công.')
    rerank_rows=[]
    for kind, display in [('lstm','LSTM'),('xlstm','xLSTM')]:
        for row in read(GA/f'reranking_{kind}.json'):
            proxy = states[kind]['trials'][row['trial_id']]['fitness']
            rerank_rows.append([display,row['trial_id'],pct(proxy),pct(row['runs'][0]['validation_macro_f1']),pct(row['runs'][1]['validation_macro_f1']),pct(row['mean_validation_macro_f1'])])
    table(['Mô hình','Trial ID','Proxy 5 ep.','Seed 17','Seed 29','TB chọn cấu hình'],rerank_rows,[3,2,2.8,2.8,2.8,3.6],9)
    p('LSTM chọn trial 17: trung bình 48,62%, cao hơn trial 6 (47,60%). xLSTM chọn trial 17: trung bình 47,88%, cao hơn trial 14 (45,82%), dù trial 14 dẫn đầu ở bước 5 epoch. Sự đảo thứ hạng này cho thấy đánh giá ngắn có thể không đại diện cho huấn luyện dài hơn.')
    picture(GA/'search_progress.png','Hình 2. Best fitness theo số trial và theo thời gian search tích lũy. Hai mô hình có cùng ngân sách trial nhưng không cùng ngân sách thời gian.',16.5)
    p('Chỉ thử 18/324 cấu hình và chỉ có một seed GA. Việc tìm được một cấu hình tốt hơn baseline là bằng chứng cho lợi ích của tìm siêu tham số trong trường hợp đó; chưa chứng minh GA tốt hơn random search. Random search là đối chứng quan trọng cho thuật toán tìm kiếm thích nghi [5].')

    # 9. GA final results and chosen settings.
    page('6.1. GA+LSTM và GA+xLSTM: kết quả cuối')
    table(['Phương pháp','Val Macro-F1','Test Accuracy','Test Macro-F1','Weighted-F1'], [[m,score(m,'validation_macro_f1'),score(m,'test_accuracy'),score(m,'test_macro_f1'),score(m,'test_weighted_f1')] for m in ORDER], [2.8,3.5,3.5,3.7,3.5])
    picture(GA/'comparison_test.png','Hình 3. Macro-F1 test trung bình ± độ lệch chuẩn mẫu giữa ba seed. Thanh sai số không phải khoảng tin cậy.',14.8)
    table(['Cấu hình chọn','GA+LSTM','GA+xLSTM'], [[k,str(selected['lstm']['genes'][k]),str(selected['xlstm']['genes'][k])] for k in ['embedding_dim','num_layers','dropout','learning_rate','weight_decay','class_weight_alpha']], [7,5,5])
    p('GA+LSTM đạt 48,32% Macro-F1, cao hơn GA+xLSTM 2,25 điểm % và cao hơn ở cả ba seed. So với cấu hình cố định của chính nó, GA+LSTM tăng 4,17 điểm %; GA+xLSTM giảm nhẹ 0,22 điểm %. Mức giảm nhỏ của trung bình xLSTM nằm trong bối cảnh dao động giữa seed lớn hơn, nên không đủ để kết luận GA có hại nói chung.')
    p('Cấu hình GA+LSTM đã tăng gấp đôi chiều embedding, giảm từ hai tầng xuống một tầng và giảm dropout. GA+xLSTM tăng chiều embedding nhưng giữ hai block, đồng thời đổi learning rate và weight decay. Vì nhiều yếu tố cùng đổi, không được gán toàn bộ cải thiện/suy giảm cho riêng một gene.')

    # 10. Class effects and anger diagnostics.
    page('7. Vì sao GA cải thiện LSTM nhưng chưa cải thiện xLSTM?')
    table(['Nhãn','GA+LSTM F1','Δ so với LSTM','GA+xLSTM F1','Δ so với xLSTM'], [[c,pct(perclass.loc[c,'GA+LSTM']),num((perclass.loc[c,'GA+LSTM']-perclass.loc[c,'LSTM'])*100),pct(perclass.loc[c,'GA+xLSTM']),num((perclass.loc[c,'GA+xLSTM']-perclass.loc[c,'xLSTM'])*100)] for c in LABELS], [3.2,3.5,3.4,3.5,3.4])
    p('GA+LSTM cải thiện F1 trung bình ở cả bảy lớp; mức tăng lớn nhất thuộc Enjoyment (+7,55 điểm %) và Fear (+5,67). Cải thiện được phân bố nhiều lớp nên đồng thời nâng Accuracy, Macro-F1 và Weighted-F1. Cấu hình mới có thể phù hợp hơn với dữ liệu nhỏ và tối ưu nhanh hơn, nhưng muốn xác nhận cần ablation độ rộng, độ sâu, dropout và learning rate.')
    h('7.1. Phân rã chênh lệch Macro-F1 của xLSTM')
    p('GA+xLSTM tăng F1 ở Enjoyment, Other, Sadness và Surprise, nhưng giảm ở Anger, Disgust và Fear. Riêng Anger giảm 11,00 điểm %: đóng góp −11,00/7 ≈ −1,57 điểm % vào Macro-F1. Sáu lớp còn lại cộng lại bù khoảng +1,35 điểm %, nên tổng thay đổi chỉ còn khoảng −0,22 điểm %. Đây là phép phân rã số học của chỉ số, không phải suy đoán.')
    p('Accuracy tăng 48,53% → 49,49% vì nó đếm số câu đúng và ưu tiên gián tiếp lớp đông mẫu. Macro-F1 cho Anger trọng số 1/7 dù lớp này chỉ có 40/693 câu test. Do đó hai chỉ số đi khác chiều là có thể giải thích bằng cơ cấu lỗi, không có mâu thuẫn tính toán.')
    diag=[]
    for method in ['xLSTM','GA+xLSTM']:
        for row in evidence['per_seed']:
            if row['method']==method:
                diag.append([method,row['seed'],f'{row["anger_correct"]}/40',pct(row['anger_precision']),pct(row['anger_recall']),pct(row['anger_f1'])])
    table(['Mô hình','Seed','Anger đúng','Precision','Recall','F1'],diag,[3.6,1.8,2.8,3,2.9,2.9],9)
    p('Recall Anger trung bình giảm 40,83% → 25,00%; nhầm Anger thành Disgust tăng 31,67% → 36,67%, thành Other tăng 12,50% → 17,50%. Cả ba seed đều giảm F1 Anger sau GA. Đây là điểm yếu cần ưu tiên kiểm tra, nhưng chưa thể quy chắc cho learning rate hay dung lượng mô hình.')

    # 11. More evidence and hypotheses.
    page('7.2. Độ dài câu, quá khớp và tính tổng quát')
    length = pd.DataFrame(evidence['length_buckets'])
    length_rows=[]
    for bucket in ['≤16','17–32','>32']:
        s=length[length.bucket_tokens_with_eos==bucket].set_index('method')
        length_rows.append([bucket,int(s.n_samples.iloc[0]),*[pct(s.loc[m,'accuracy_mean']) for m in ORDER]])
    table(['Token có EOS','Số câu test','LSTM Acc.','GA+LSTM Acc.','xLSTM Acc.','GA+xLSTM Acc.'],length_rows,[2.9,2.3,2.8,3,2.8,3.2],9)
    p('Phân tích theo độ dài chỉ là mô tả hậu nghiệm. Nhóm >32 token có 75 câu và thành phần nhãn khác các nhóm còn lại. xLSTM cố định không hơn LSTM trong nhóm này (44,89% so với 45,78% Accuracy); GA+xLSTM chỉ nhỉnh hơn GA+LSTM một ít (44,44% so với 43,56%). Chưa có cơ sở kết luận khả năng nhớ chuỗi dài là nguyên nhân của khác biệt tổng thể.')
    h('Dấu hiệu từ đường học')
    table(['Run minh họa','Loss train: epoch chọn → cuối','Val Macro-F1: tốt nhất → cuối'], [
        ['GA+LSTM, seed 42','0,453 → 0,092','47,64% → 44,67%'],
        ['GA+xLSTM, seed 42','0,506 → 0,323','47,99% → 41,93%'],
        ['GA+xLSTM, seed 2026','0,520 → 0,310','47,45% → 43,67%']
    ], [5.1,5.9,6])
    p('Loss train tiếp tục giảm trong khi Macro-F1 validation giảm sau epoch tốt nhất: mô hình khớp train tốt hơn nhưng không đồng thời cải thiện tiêu chí ngoài train. Mẫu hình này phù hợp nguy cơ quá khớp hoặc lệch giữa mục tiêu loss và Macro-F1. Early stopping đã hạn chế tác hại bằng cách dùng checkpoint tốt nhất, không dùng epoch cuối.')
    p('Không so trực tiếp trị số loss giữa hai phương pháp để kết luận mô hình nào tốt hơn: α, dropout và phân phối lỗi khác nhau. Chưa đo train F1/validation loss ở chế độ eval, nên chưa lượng hóa đầy đủ khoảng cách tổng quát hóa bằng cùng một chỉ số.')
    h('Các giả thuyết hợp lý nhưng chưa được chứng minh')
    p('Thứ nhất, corpus nhỏ và embedding học từ đầu có thể thiếu tri thức ngữ nghĩa cho teencode, phủ định hoặc sắc thái gần nhau. Điểm UNK rất thấp không có nghĩa biểu diễn đã hiểu tốt ngữ nghĩa. Thứ hai, proxy 5 epoch và một seed GA có thể ưu tiên cấu hình học nhanh hoặc phù hợp một validation cụ thể. Thứ ba, tăng chiều mô hình và learning rate có thể thay đổi cân bằng lỗi giữa các lớp.')
    note('Validation của GA+xLSTM tăng 44,48% → 47,81%, nhưng test Macro-F1 không tăng. Đây là biểu hiện lựa chọn theo validation không chuyển thành lợi ích test trong lần chạy này; chưa đủ để khẳng định có dịch chuyển miền, nhãn sai hoặc rò rỉ dữ liệu.')

    # 12. Synthesis and conclusion.
    page('8. Tổng hợp ưu nhược điểm và kết luận')
    table(['Phương pháp','Điểm mạnh quan sát','Điểm yếu / đánh đổi'], [
        ['LSTM','Chi phí thấp, triển khai đơn giản; F1 Other/Sadness tốt hơn xLSTM cố định.','Macro-F1 test thấp nhất trong bốn cấu hình; dao động seed lớn ở cấu hình ban đầu.'],
        ['xLSTM','Tốt hơn LSTM cố định ở Anger/Fear/Enjoyment; dao động F1 tổng nhỏ trong ba seed.','Nhiều bộ nhớ trung gian và thời gian hơn; chưa có lợi thế rõ theo độ dài câu.'],
        ['GA+LSTM','Macro-F1, Accuracy và Weighted-F1 trung bình cao nhất; tăng F1 cả bảy lớp so với LSTM.','Thêm chi phí tìm kiếm; tổng tham số tăng gần gấp đôi; chưa xác nhận trên test mới.'],
        ['GA+xLSTM','Tăng Accuracy và F1 Other/Surprise so với xLSTM cố định; có kết quả tốt hơn GA+LSTM ở Surprise.','F1 Anger giảm; Macro-F1 trung bình không tăng; chi phí pipeline và dao động seed cao hơn.']
    ], [3,6.7,7.3])
    table(['Kiến trúc GA','Search (s)','Chọn cấu hình (s)','Train cuối (s)','Tổng (phút)'], [[m,num(costs.loc[k,'search_seconds'],1),num(costs.loc[k,'rerank_seconds'],1),num(costs.loc[k,'final_training_seconds'],1),num(costs.loc[k,'total_seconds']/60,2)] for k,m in [('lstm','LSTM'),('xlstm','xLSTM')]], [3.3,3,3.9,3.3,3.5])
    p('Pipeline GA+xLSTM mất khoảng 16,15 phút so với 4,60 phút cho GA+LSTM, tức khoảng 3,51 lần. Sau chọn cấu hình, GA+LSTM có 497.415 tham số, GA+xLSTM có 584.087. Chi phí trên chưa gồm tiền xử lý, test và sinh báo cáo; không đồng nhất với thời gian người dùng chờ toàn bộ phiên làm việc.')
    h('Kết luận')
    p('Trong phạm vi cấu hình và dữ liệu đã chạy, GA+LSTM là phương án tốt nhất nếu ưu tiên Macro-F1 test và chi phí tính toán. xLSTM cấu hình cố định nhỉnh hơn LSTM cố định, nhưng lợi thế đảo chiều sau tìm siêu tham số: kết quả phụ thuộc mạnh vào cấu hình và quy trình chọn, không chỉ tên kiến trúc.')
    p('GA không tạo lợi ích đồng đều cho cả hai kiến trúc. Cải thiện của GA+LSTM lan tỏa qua nhiều lớp; với GA+xLSTM, lợi ích ở một số lớp bị suy giảm Anger bù trừ. Không có bằng chứng đủ mạnh để kết luận mọi xLSTM kém LSTM hoặc GA nói chung không phù hợp xLSTM.')
    p('Baseline TF-IDF + LinearSVC hiện có đạt 53,19% Macro-F1 validation, vẫn cao hơn GA+LSTM 49,22% và GA+xLSTM 47,81%. Đây là mốc cảnh báo cần giữ đối chứng đơn giản; chưa có test của TF-IDF trong gói thực nghiệm này, nên không xếp hạng nó bằng điểm test.')
    note('Giới hạn chính: ba seed cuối và một seed GA; seed 42 còn được dùng ở proxy; test đã được xem từ giai đoạn trước; không có random search đối chứng, ablation hay kiểm định ý nghĩa. Độ lệch chuẩn giữa seed không thay thế đánh giá trên mẫu dữ liệu độc lập.')

    # 13. Actionable improvement plan.
    page('9. Hướng cải thiện và kế hoạch kiểm chứng')
    p('Ưu tiên các thí nghiệm có thể trả lời một giả thuyết rõ ràng. Mọi lựa chọn mới phải dùng train/validation hoặc các fold nội bộ; cần test mới hoặc đánh giá ngoài để xác nhận, thay vì tiếp tục tối ưu theo bảng test hiện tại.')
    table(['Ưu tiên','Hướng thực hiện','Giả thuyết / tiêu chí kiểm chứng'], [
        ['1. Xác nhận lợi ích GA','Thêm random search với đúng 18 trial, cùng proxy và bước chạy lại; lặp nhiều seed tìm kiếm. Báo kết quả cả theo số trial và thời gian GPU.','GA có hơn tìm ngẫu nhiên hay chỉ hơn cấu hình mặc định? Dùng mean/SD qua các lần tìm kiếm, không chỉ seed train [5].'],
        ['2. Tách nguyên nhân','Ablation chiều 64/128, tầng 1/2, LR 0,001/0,003, dropout, α; giữ các yếu tố còn lại. Có bảng gần khớp số tham số.','Yếu tố nào cải thiện LSTM hoặc làm giảm Anger của xLSTM? So Macro-F1 và F1 từng lớp trên validation theo cặp seed.'],
        ['3. Cải thiện biểu diễn','Thử embedding tiếng Việt tiền huấn luyện hoặc một baseline PhoBERT. Với fastText, thiết kế ánh xạ token/đơn vị từ phù hợp; với PhoBERT dùng tiền xử lý riêng theo tài liệu.','Tri thức ngôn ngữ ngoài corpus có giúp câu ít dữ liệu/teencode không? Giữ bảng riêng vì tài nguyên tiền huấn luyện khác [6–8].'],
        ['4. Sửa điểm yếu lớp Anger','Rà soát lỗi Anger–Disgust và Anger–Other trên validation; thử α, sampling theo lớp hoặc focal loss, từng thay đổi riêng.','Tăng recall Anger có làm precision hay lớp khác giảm không? Đánh giá đồng thời P/R/F1 từng lớp; không chỉ Accuracy.'],
        ['5. Cải thiện chọn cấu hình','Tăng budget proxy theo nhiều mức, hoặc chạy lại top nhiều hơn; đưa seed và độ ổn định vào bước chọn; giữ tổng ngân sách rõ ràng.','Giảm sai lệch do fitness 5 epoch. Theo dõi việc đảo thứ hạng ứng viên khi tăng epoch.'],
        ['6. Encoder và chi phí','Thử BiLSTM, pooling có attention hoặc tổ hợp mLSTM/sLSTM trong bảng riêng. Đo profiler, inference và VRAM; tối ưu backend sau khi kiểm tra số học.','Cải thiện biểu diễn câu hoặc hiệu năng thực thi có thực sự chuyển thành lợi ích? So cùng ngân sách và kiểm tra bất biến padding.']
    ], [3.3,7,6.7],9.1)
    p('Đề xuất bước gần nhất: giữ GA+LSTM làm mốc, chạy random search đối chứng và ablation LR/chiều/tầng trước; song song rà lỗi Anger của xLSTM trên validation. Sau khi xác định yếu tố có ích, mới mở rộng tìm kiếm hoặc thêm mô hình tiền huấn luyện. Không cần tăng độ phức tạp đồng loạt.')
    p('Với điều kiện GPU 4 GB, mô hình tiền huấn luyện cần được kiểm tra bộ nhớ bằng batch nhỏ trước. Đây là hướng đề xuất, chưa có kết quả hoặc cam kết cải thiện trong báo cáo. Khi muốn đánh giá hiệu quả triển khai, đo latency, throughput và bộ nhớ inference riêng; số giây train hiện tại không thay thế các chỉ số đó.')

    # 14. References and provenance.
    page('10. Tài liệu tham khảo và nguồn bằng chứng')
    refs = [
        ('[1] Ho, V. A. và cộng sự (2019/2020). Emotion Recognition for Vietnamese Social Media Text. PACLING 2019; arXiv v2 năm 2020.', 'https://arxiv.org/abs/1911.09339'),
        ('[2] Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation, 9(8), 1735–1780.', 'https://doi.org/10.1162/neco.1997.9.8.1735'),
        ('[3] Beck, M. và cộng sự (2024). xLSTM: Extended Long Short-Term Memory. NeurIPS 2024; arXiv:2405.04517.', 'https://arxiv.org/abs/2405.04517'),
        ('[4] NX-AI. Mã nguồn xLSTM; commit ab22eadbd245f293dd8dec38ed29963d73758a12. Bản tải về và thay đổi lazy-import được lưu trong vendor/.', 'https://github.com/NX-AI/xlstm/tree/ab22eadbd245f293dd8dec38ed29963d73758a12'),
        ('[5] Bergstra, J. & Bengio, Y. (2012). Random Search for Hyper-Parameter Optimization. JMLR, 13, 281–305.', 'https://www.jmlr.org/papers/v13/bergstra12a.html'),
        ('[6] Bojanowski, P., Grave, E., Joulin, A. & Mikolov, T. (2017). Enriching Word Vectors with Subword Information. TACL, 5, 135–146.', 'https://aclanthology.org/Q17-1010/'),
        ('[7] Nguyen, D. Q. & Nguyen, A. T. (2020). PhoBERT: Pre-trained language models for Vietnamese. Findings of EMNLP 2020, 1037–1042.', 'https://aclanthology.org/2020.findings-emnlp.92/'),
        ('[8] VinAIResearch. PhoBERT: hướng dẫn mô hình và tiền xử lý; yêu cầu phân đoạn từ đối với đầu vào PhoBERT được nêu trong tài liệu.', 'https://github.com/VinAIResearch/PhoBERT')
    ]
    for title,url in refs:
        p(title)
        para=p(); add_link(para,url,url)
        para.paragraph_format.space_after=Pt(7)
    h('Nguồn nội bộ và khả năng kiểm tra')
    p('Các số liệu thực nghiệm đến từ file CSV/JSON trong results/lstm_xlstm_comparison/ và results/ga_lstm_xlstm_comparison/, không lấy từ điểm benchmark của các bài báo. Tài liệu bên ngoài hỗ trợ mô tả kiến trúc và hướng nghiên cứu, không chứng minh nguyên nhân của các chênh lệch hiện tại.')
    p('Phân tích hậu nghiệm trong reports/analysis_evidence.json, accuracy_by_length.csv và anger_diagnostics.csv được tạo từ các dự đoán đã lưu. Script scripts/create_docx_report.py tạo báo cáo; reports/docx_build_manifest.json ghi hash tài liệu và các nguồn dữ liệu chính.')
    p('Các đường dẫn trong báo cáo tính từ thư mục D:\\UIT_VSMEC. Tài liệu tham khảo được đối chiếu ngày 28/10/2026. Không bổ sung số liệu từ mô hình chưa chạy hoặc suy đoán kết quả của các hướng cải thiện.')

    # 15. Detailed runs and reproducibility.
    page('Phụ lục. Kết quả từng seed và tệp tái lập')
    rows=[]
    for method in ORDER:
        for _,r in raw[raw.method==method].sort_values('seed').iterrows():
            rows.append([method,int(r.seed),f'{int(r.best_epoch)}/{int(r.epochs_run)}',pct(r.validation_macro_f1),pct(r.test_accuracy),pct(r.test_macro_f1)])
    table(['Phương pháp','Seed','Epoch chọn/chạy','Val F1','Test Acc.','Test F1'],rows,[3.4,1.7,3.2,2.9,2.9,2.9],9)
    table(['Tệp hoặc thư mục','Nội dung'], [
        ['configs/lstm_xlstm_comparison.json','Thiết lập baseline cố định.'],
        ['configs/ga_lstm_xlstm_comparison.json','Không gian và ngân sách GA thực chạy 18 trial/mô hình.'],
        ['results/ga_lstm_xlstm_comparison/locked_protocol.json','Cấu hình khóa, phiên bản thư viện và hash mã nguồn/dữ liệu.'],
        ['…/search/<model>/search_state.json','Cấu hình, fitness, cha mẹ, đột biến và quần thể từng thế hệ.'],
        ['…/reranking_<model>.json; selected_configs.json','Chạy lại top 2 và lựa chọn cấu hình cuối.'],
        ['…/final/<model>_seed<seed>/','Checkpoint best.pt, history.csv, dự đoán và metrics validation/test.'],
        ['…/verification/verification.json','Kết quả chấm lại 56 bộ dự đoán và kiểm tra ngân sách, elitism, cấu hình, epoch.']
    ], [9,8],8.8)
    p('Dấu “…” trong bảng chỉ results/ga_lstm_xlstm_comparison. Hướng dẫn chạy lại và tiếp tục run gián đoạn nằm trong GA_EXPERIMENTS.md; hướng dẫn baseline trong EXPERIMENTS.md. Thư mục kết quả cũ được giữ để đối chiếu, không thay thế bằng một seed tốt nhất.')
    note('Tóm tắt cuối: ưu tiên GA+LSTM làm mốc thực dụng ở thời điểm này; tiếp tục nghiên cứu xLSTM bằng đối chứng tìm kiếm, ablation và phân tích lỗi lớp Anger. Xác nhận mọi cải thiện mới trên dữ liệu đánh giá độc lập.')


if __name__ == '__main__':
    build()

