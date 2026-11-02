"""Produce Vietnamese report and figures from completed, paired experiment runs."""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, default=ROOT / 'results/lstm_xlstm_comparison')
    args = parser.parse_args()
    out = args.outdir
    runs = pd.read_csv(out / 'summary_runs.csv')
    cfg = json.loads((out / 'locked_config.json').read_text(encoding='utf-8'))
    env = json.loads((out / 'environment.json').read_text(encoding='utf-8'))
    has_test = 'test_macro_f1' in runs
    split = 'test' if has_test else 'validation'
    fields = ['validation_macro_f1'] + (['test_accuracy', 'test_macro_f1', 'test_weighted_f1'] if has_test else [])
    aggregate = runs.groupby('model')[fields].agg(['mean', 'std'])
    flat = aggregate.copy()
    flat.columns = ['_'.join(c) for c in flat.columns]
    flat.to_csv(out / 'summary_mean_std.csv')
    local_date = datetime.fromisoformat(env['started_utc']).astimezone(timezone(timedelta(hours=7))).strftime('%d/%m/%Y')
    lines = ['# Thực nghiệm LSTM và xLSTM trên UIT-VSMEC', '',
             f'Ngày thực nghiệm: {local_date}. Điểm trong bảng tính theo %, độ lệch chuẩn mẫu giữa các seed (ddof=1).', '',
             '## Khác biệt kiến trúc', '',
             'LSTM chuẩn giữ trạng thái ô nhớ dạng vector và điều tiết thông tin bằng các cổng. Biến thể mLSTM trong họ xLSTM dùng bộ nhớ ma trận, cổng hàm mũ được ổn định và được ghép thành các block residual. Backend của thí nghiệm tính mLSTM theo dạng song song bằng PyTorch; khả năng song song về thuật toán không bảo đảm nhanh hơn LSTM dùng cuDNN trên mô hình nhỏ. [Nguồn kiến trúc: Beck et al., 2024](https://arxiv.org/abs/2405.04517).', '',
             'Thí nghiệm này đo khả năng phân loại bảy cảm xúc trên các bình luận ngắn, không phải bài kiểm tra chuyên biệt về ghi nhớ chuỗi dài.', '',
             '## Thiết kế', '',
             '- Train 5.547 mẫu (đã bỏ mẫu trùng validation), validation 686 mẫu, test chính thức 693 mẫu.',
             f'- Seed cố định: {cfg["seeds"]}. Không chọn seed tốt nhất; không tìm siêu tham số.',
             f'- SentencePiece unigram: yêu cầu tối đa {cfg["vocab_size"]} token, vocabulary thực tế {env["actual_vocab_size"]}; học chỉ từ train; NFC, giữ dấu, chữ hoa, dấu câu và emoji; EOS, PAD bên phải; tối đa {cfg["max_length"]} token.',
             f'- Cả hai: embedding học từ đầu {cfg["embedding_dim"]} chiều (khởi tạo giống nhau theo seed), {cfg["num_layers"]} tầng, encoder một chiều, masked mean pooling, dropout {cfg["dropout"]}.',
             f'- LSTM: torch.nn.LSTM + LayerNorm. xLSTM: xLSTMBlockStack chính thức với {cfg["num_layers"]} block mLSTM, 4 head, convolution kernel 4, không có sLSTM.',
             f'- AdamW, LR {cfg["learning_rate"]}, batch {cfg["batch_size"]}, weight decay {cfg["weight_decay"]} (bỏ bias/norm), gradient clip {cfg["gradient_clip"]}, FP32; trọng số CE là (N/(7*n_c))^alpha với alpha={cfg["class_weight_alpha"]}, chuẩn hóa trung bình 1.',
             f'- Tối đa {cfg["max_epochs"]} epoch, patience {cfg["patience"]}; lưu checkpoint theo Macro-F1 validation cao nhất (hòa giữ epoch sớm hơn). Chỉ đọc test để đánh giá sau khi tất cả lần train hoàn tất.',
             f'- Thiết bị: {env["gpu"] or env["device"]}; PyTorch {env["packages"]["torch"]}.', '',
             '## Kết quả tổng hợp', '', '| Mô hình | Validation Macro-F1 |' + (' Test Accuracy | Test Macro-F1 | Test Weighted-F1 |' if has_test else ''),
             '|---|' + '---:|' * len(fields)]
    for model in cfg['models']:
        cells = [f'{100*aggregate.loc[model,(field,"mean")]:.2f} ± {100*aggregate.loc[model,(field,"std")]:.2f}' for field in fields]
        lines.append('| ' + ('xLSTM (mLSTM-only)' if model == 'xlstm' else 'LSTM') + ' | ' + ' | '.join(cells) + ' |')
    paired = runs.pivot(index='seed', columns='model', values=f'{split}_macro_f1')
    delta = 100 * (paired.xlstm - paired.lstm)
    lines += ['', f'Chênh lệch Macro-F1 {split} (xLSTM − LSTM): **{delta.mean():+.2f} điểm phần trăm**, độ lệch chuẩn giữa các cặp seed {delta.std():.2f}.', '',
              'Đây là so sánh cấu hình cố định với ba seed, không phải bằng chứng một họ kiến trúc luôn tốt hơn. Độ lệch chuẩn giữa seed không phải khoảng tin cậy trên quần thể dữ liệu.', '',
              '## Từng lần chạy', '', '| Mô hình | Seed | Epoch tốt nhất / đã chạy | Val F1 | ' + ('Test F1 | ' if has_test else '') + 'Thời gian train (giây) |',
              '|---|---:|---:|---:|' + ('---:|' if has_test else '') + '---:|']
    for _, r in runs.iterrows():
        test_cell = f' {100*r.test_macro_f1:.2f} |' if has_test else ''
        lines.append(f'| {r.model} | {r.seed} | {r.best_epoch} / {r.epochs_run} | {100*r.validation_macro_f1:.2f} |{test_cell} {r.training_seconds_including_validation_and_checkpoints:.1f} |')
    lines += ['', '## Chi phí và giới hạn so sánh', '', '| Mô hình | Tổng tham số | Tham số ngoài embedding | Thời gian train TB (giây) | Peak GPU allocated TB (MiB) |', '|---|---:|---:|---:|---:|']
    for model, group in runs.groupby('model'):
        lines.append(f'| {model} | {int(group.parameters.iloc[0]):,} | {int(group.non_embedding_parameters.iloc[0]):,} | {group.training_seconds_including_validation_and_checkpoints.mean():.1f} | {group.peak_cuda_allocated_mb.mean():.1f} |')
    lines += ['', 'Thời gian bao gồm huấn luyện, validation và ghi checkpoint; không gồm huấn luyện tokenizer. Peak GPU allocated là bộ nhớ tensor PyTorch, không phải toàn bộ VRAM của tiến trình. Đây không phải phép đo tốc độ inference.', '',
              'Hai mô hình có cùng chiều embedding, số tầng và ngân sách epoch, nhưng không khớp tuyệt đối số tham số hoặc chi phí tính toán. Số epoch thực tế có thể khác do dừng sớm. Dropout và cấu trúc bên trong block khác nhau. Kết quả phản ánh toàn bộ hai cấu hình này, không tách riêng tác động của bộ nhớ ma trận.', '',
              'Đây là xLSTM chỉ gồm mLSTM, không đại diện cho mọi tổ hợp mLSTM/sLSTM và không sử dụng mô hình xLSTM 7B hay embedding tiền huấn luyện. Chưa chạy GA hoặc so sánh sau tối ưu siêu tham số.', '',
              'Đã cố định seed và bật deterministic ở chế độ warn_only: CUDA cumsum trong mLSTM không có kernel deterministic ở PyTorch 2.5.1, nên không cam kết tái lập chính xác từng bit. Mã nguồn có một sửa đổi lazy-import cho sLSTM CUDA để chạy khi thiếu CUDA Toolkit; không sửa phép tính mLSTM (xem vendor/README.md).', '',
              '## F1 từng lớp', '', '| Nhãn | LSTM | xLSTM | Chênh lệch (điểm %) |', '|---|---:|---:|---:|---:|']
    class_rows, matrices = [], {}
    for model in cfg['models']:
        records = [json.loads((out / f'{model}_seed{seed}' / f'metrics_{split}.json').read_text(encoding='utf-8')) for seed in cfg['seeds']]
        names = records[0]['confusion_matrix_labels']
        matrices[model] = np.mean([r['confusion_matrix_rows_true_columns_predicted'] for r in records], axis=0)
        for label in names:
            vals = [r['per_class'][label]['f1-score'] for r in records]
            class_rows.append({'model': model, 'label': label, 'f1_mean': np.mean(vals), 'f1_std': np.std(vals, ddof=1)})
    per_class = pd.DataFrame(class_rows)
    per_class.to_csv(out / f'per_class_{split}.csv', index=False)
    pivot = per_class.pivot(index='label', columns='model', values='f1_mean').loc[names]
    for label, row in pivot.iterrows():
        lines.append(f'| {label} | {row.lstm*100:.2f} | {row.xlstm*100:.2f} | {(row.xlstm-row.lstm)*100:+.2f} |')
    differences = (pivot.xlstm - pivot.lstm) * 100
    runtime = runs.groupby('model').training_seconds_including_validation_and_checkpoints.mean()
    lines += ['', '## Khác biệt quan sát được', '',
              f'- xLSTM đạt Macro-F1 {split} cao hơn LSTM ở {int((delta > 0).sum())}/{len(delta)} cặp seed.',
              f'- Lớp có chênh lệch xLSTM − LSTM cao nhất: {differences.idxmax()} ({differences.max():+.2f} điểm %); thấp nhất: {differences.idxmin()} ({differences.min():+.2f} điểm %).',
              f'- Thời gian train trung bình của xLSTM bằng {runtime.xlstm/runtime.lstm:.2f} lần LSTM trên thiết bị và backend này; đây không phải so sánh các kernel đã tối ưu ngang nhau.', '']
    gold = pd.read_csv(ROOT / cfg[split], keep_default_na=False)[['sample_id', 'text', 'label']]
    paired_errors = gold.copy()
    for model in cfg['models']:
        cols = []
        for seed in cfg['seeds']:
            col = f'{model}_seed{seed}'
            pred = pd.read_csv(out / f'{model}_seed{seed}/predictions_{split}.csv')
            paired_errors = paired_errors.merge(pred.rename(columns={'pred_label': col}), on='sample_id', validate='one_to_one')
            cols.append(col)
        paired_errors[f'{model}_correct_seeds'] = paired_errors[cols].eq(paired_errors.label, axis=0).sum(axis=1)
    paired_errors.to_csv(out / f'paired_errors_{split}.csv', index=False, encoding='utf-8-sig')
    lines += [f'File `paired_errors_{split}.csv` chứa toàn bộ văn bản, nhãn thật, dự đoán của từng seed và số seed dự đoán đúng, giúp phân tích lỗi mà không chỉ chọn ví dụ có lợi cho một mô hình.', '']
    baseline = json.loads((ROOT / 'results/baseline_validation/metrics_validation.json').read_text(encoding='utf-8'))
    lines += ['', f'Baseline TF-IDF + LinearSVC đã có đạt **{baseline["macro_f1"]*100:.2f}% Macro-F1 validation**. Chỉ đối chiếu cột validation; baseline này chưa được đánh giá test trong thực nghiệm hiện tại.', '',
              '## Tiền xử lý', '',
              'Vocabulary thực tế có thể nhỏ hơn 4.000 vì hard_vocab_limit=False. Thống kê dưới đây dùng tokenizer đã học trên train:', '',
              '| Tập | Mẫu | Mẫu bị cắt ngắn | Token UNK / token trước cắt |', '|---|---:|---:|---:|']
    for name, stat in env['tokenization'].items():
        lines.append(f'| {name} | {stat["n_samples"]} | {stat["truncated"]} | {stat["unknown_tokens"]} / {stat["tokens_before_truncation"]} |')
    lines += ['', '## Biểu đồ', '', '![Đường học](learning_curves.png)', '', f'![F1 từng lớp](per_class_{split}.png)', '', f'![Ma trận nhầm lẫn](confusion_{split}.png)', '',
              '## Tái lập và nguồn', '',
              'Từ thư mục dự án: `python scripts/check_sequence_models.py`, sau đó `python scripts/compare_lstm_xlstm.py --outdir results/new_comparison` và `python scripts/report_lstm_xlstm.py --outdir results/new_comparison`.', '',
              'Mỗi run lưu best.pt, history.csv, predictions/metrics validation và test; environment.json lưu phiên bản, SHA-256 dữ liệu, mã huấn luyện và tokenizer. locked_config.json ghi cấu hình trước khi train. Script từ chối ghi đè thư mục kết quả có dữ liệu.', '',
              'Kiểm tra độc lập bằng bộ đánh giá có sẵn: `python scripts/verify_sequence_experiment.py --outdir results/new_comparison`. Kết quả kiểm tra lưu trong verification/verification.json.', '',
              '- [xLSTM — bài báo NeurIPS 2024](https://arxiv.org/abs/2405.04517)',
              f'- [NX-AI/xlstm — commit đã pin](https://github.com/NX-AI/xlstm/tree/{cfg["xlstm_source_commit"]})',
              '- [UIT-VSMEC — Ho et al.](https://arxiv.org/abs/1911.09339)', '']
    (out / 'REPORT_VI.md').write_text('\n'.join(lines), encoding='utf-8')
    plt.rcParams.update({'figure.dpi': 140, 'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, model in zip(axes, cfg['models']):
        for seed in cfg['seeds']:
            h = pd.read_csv(out / f'{model}_seed{seed}/history.csv')
            ax.plot(h.epoch, h.validation_macro_f1 * 100, marker='.', label=f'seed {seed}')
        ax.set(title=model, xlabel='Epoch', ylabel='Validation Macro-F1 (%)')
        ax.legend()
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / 'learning_curves.png')
    plt.close(fig)
    ax = (pivot * 100).plot.bar(figsize=(10, 4), rot=0, ylabel=f'{split.title()} F1 (%)')
    ax.figure.tight_layout()
    ax.figure.savefig(out / f'per_class_{split}.png')
    plt.close(ax.figure)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, model in zip(axes, cfg['models']):
        cm = matrices[model]
        cm = cm / cm.sum(axis=1, keepdims=True) * 100
        ax.imshow(cm, vmin=0, vmax=100, cmap='Blues')
        ax.set(xticks=range(7), yticks=range(7), xticklabels=names, yticklabels=names,
               title=f'{model}: mean row-normalized {split} confusion (%)', xlabel='Predicted', ylabel='True')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        for i in range(7):
            for j in range(7):
                ax.text(j, i, f'{cm[i,j]:.0f}', ha='center', va='center', color='white' if cm[i,j] > 50 else 'black')
    fig.tight_layout()
    fig.savefig(out / f'confusion_{split}.png')
    plt.close(fig)
    print(out / 'REPORT_VI.md')


if __name__ == '__main__':
    main()
