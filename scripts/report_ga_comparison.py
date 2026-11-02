"""Report GA-LSTM / GA-xLSTM against the preserved fixed baselines."""
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


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, default=ROOT / 'results/ga_lstm_xlstm_comparison')
    args = parser.parse_args()
    out = args.outdir
    lock = read(out / 'locked_protocol.json')
    spec, base = lock['spec'], lock['base_training_config']
    baseline = ROOT / spec['baseline_directory']
    ga_runs = pd.read_csv(out / 'summary_runs.csv')
    fixed = pd.read_csv(baseline / 'summary_runs.csv')
    fixed['method'] = fixed.model.str.upper()
    ga_runs['method'] = 'GA+' + ga_runs.model.str.upper()
    all_runs = pd.concat([fixed, ga_runs], ignore_index=True)
    all_runs.to_csv(out / 'comparison_runs.csv', index=False)
    order = ['LSTM', 'GA+LSTM', 'XLSTM', 'GA+XLSTM']
    fields = ['validation_macro_f1', 'test_accuracy', 'test_macro_f1', 'test_weighted_f1']
    summary = all_runs.groupby('method')[fields].agg(['mean', 'std']).loc[order]
    flat = summary.copy()
    flat.columns = ['_'.join(c) for c in flat.columns]
    flat.to_csv(out / 'comparison_mean_std.csv')
    selected = read(out / 'selected_configs.json')['selected']
    states = {kind: read(out / 'search' / kind / 'search_state.json') for kind in spec['models']}
    env = read(out / 'environment.json')
    day = datetime.fromisoformat(env['started_utc']).astimezone(timezone(timedelta(hours=7))).strftime('%d/%m/%Y')
    lines = ['# GA + LSTM và GA + xLSTM trên UIT-VSMEC', '', f'Ngày chạy: {day}. Điểm là phần trăm, trung bình ± độ lệch chuẩn mẫu giữa các seed.', '',
        '## Phương pháp', '',
        'GA tìm siêu tham số; AdamW học trọng số mạng. LSTM và xLSTM giữ nguyên mã mô hình của thực nghiệm trước. xLSTM ở đây chỉ gồm các block mLSTM, không có sLSTM.', '',
        f'- Dữ liệu: {env["tokenization"]["train"]["n_samples"]} train, {env["tokenization"]["validation"]["n_samples"]} validation, {env["tokenization"]["test"]["n_samples"]} test; dùng lại chính xác tokenizer train-only đã lưu của baseline.',
        f'- Mỗi kiến trúc: quần thể {spec["ga"]["population_size"]}, {spec["ga"]["offspring_generations"]} thế hệ sinh thêm, giữ {spec["ga"]["elites"]} elite. Tổng {spec["ga"]["unique_trial_budget"]} cấu hình duy nhất, mỗi cấu hình {spec["ga"]["proxy_epochs"]} epoch; fitness = Macro-F1 validation tốt nhất.',
        f'- Tournament {spec["ga"]["tournament_size"]}, uniform crossover xác suất {spec["ga"]["crossover_probability"]}, đột biến mỗi gene xác suất {spec["ga"]["mutation_probability_per_gene"]:.4f}. Loại đề xuất trùng, không tính lại fitness; trial lỗi vẫn chiếm ngân sách và xếp cuối.',
        f'- Seed tìm kiếm {spec["ga"]["search_seed"]}, seed train proxy {spec["ga"]["training_seed"]}. Hai kiến trúc dùng cùng quần thể ban đầu, có cấu hình baseline; các thế hệ sau khác nhau theo fitness.',
        f'- Chạy lại top {spec["rerank"]["top_k"]} cấu hình trên seed {spec["rerank"]["seeds"]}, tối đa {spec["rerank"]["max_epochs"]} epoch, patience {spec["rerank"]["patience"]}; chọn theo Macro-F1 validation trung bình. Nếu hòa chính xác, chọn trial ID nhỏ hơn.',
        f'- Khóa cấu hình rồi train từ đầu với seed {spec["final"]["seeds"]}, tối đa {spec["final"]["max_epochs"]} epoch, patience {spec["final"]["patience"]}; chọn checkpoint bằng validation, không chọn seed tốt nhất.',
        '- Chỉ đọc test sau khi đã lưu lựa chọn của cả hai kiến trúc và hoàn tất cả sáu lần train cuối. Cấu hình baseline được giữ nguyên từ thực nghiệm trước.',
        f'- Thiết bị: {env["gpu"]}; FP32. Batch {base["batch_size"]}, độ dài tối đa {base["max_length"]}, masked mean pooling; không dùng embedding tiền huấn luyện.', '',
        '## Không gian tìm kiếm và cấu hình được chọn', '',
        '| Siêu tham số | Không gian chung | GA+LSTM | GA+xLSTM |', '|---|---|---:|---:|']
    for name, choices in spec['search_space'].items():
        lines.append(f'| {name} | {choices} | {selected["lstm"]["genes"][name]} | {selected["xlstm"]["genes"][name]} |')
    lines += ['', 'Trial ID bắt đầu từ 0. Lựa chọn sau bước chạy lại:', '']
    for kind in spec['models']:
        lines.append(f'- {kind}: trial {selected[kind]["trial_id"]}, Macro-F1 validation trung bình ở bước chọn cấu hình = {selected[kind]["mean_validation_macro_f1"]*100:.2f}%.')
    lines += ['', '## Kết quả cuối', '', '| Phương pháp | Val Macro-F1 | Test Accuracy | Test Macro-F1 | Test Weighted-F1 |', '|---|---:|---:|---:|---:|']
    for method in order:
        cells = [f'{summary.loc[method,(field,"mean")]*100:.2f} ± {summary.loc[method,(field,"std")]*100:.2f}' for field in fields]
        lines.append('| ' + method + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', 'Sáu lần train cuối sau GA:', '',
              '| Mô hình | Seed | Epoch chọn / đã chạy | Val Macro-F1 | Test Macro-F1 | Test Accuracy |',
              '|---|---:|---:|---:|---:|---:|']
    for _, row in ga_runs.iterrows():
        lines.append(f'| {row.method} | {row.seed} | {row.best_epoch} / {row.epochs_run} | {row.validation_macro_f1*100:.2f} | {row.test_macro_f1*100:.2f} | {row.test_accuracy*100:.2f} |')
    lines += ['', '## Chênh lệch với cấu hình cố định', '']
    deltas = []
    for kind in spec['models']:
        fixed_model = fixed[fixed.model == kind].set_index('seed')
        ga_model = ga_runs[ga_runs.model == kind].set_index('seed')
        if set(fixed_model.index) != set(ga_model.index):
            raise ValueError('Final and baseline seeds must match for paired comparison')
        delta = (ga_model.test_macro_f1 - fixed_model.test_macro_f1) * 100
        lines.append(f'- GA+{kind.upper()} − {kind.upper()}: **{delta.mean():+.2f} điểm phần trăm Macro-F1 test**, cao hơn ở {int((delta > 0).sum())}/{len(delta)} seed; độ lệch chuẩn của chênh lệch {delta.std():.2f} điểm %.')
        for seed, value in delta.items():
            deltas.append({'model': kind, 'seed': seed, 'test_macro_f1_delta_percentage_points': value})
    between = ga_runs.pivot(index='seed', columns='model', values='test_macro_f1')
    gap = (between.xlstm - between.lstm) * 100
    lines += ['', f'GA+xLSTM − GA+LSTM: **{gap.mean():+.2f} điểm phần trăm Macro-F1 test**; GA+xLSTM cao hơn ở {int((gap > 0).sum())}/{len(gap)} seed.', '',
              'Các chênh lệch trên là mô tả kết quả, không phải kiểm định ý nghĩa thống kê.', '',
              '## Chi phí tìm kiếm và huấn luyện', '',
              '| Kiến trúc | Trial thành công / tổng | Search (giây) | Chạy lại chọn cấu hình (giây) | Train cuối 3 seed (giây) | Tổng (giây) |',
              '|---|---:|---:|---:|---:|---:|']
    pd.DataFrame(deltas).to_csv(out / 'paired_deltas.csv', index=False)
    costs = []
    search_rows = []
    for kind in spec['models']:
        trials = states[kind]['trials']
        search_time = sum(r['elapsed_seconds'] for r in trials)
        rerank_time = sum(r['training_seconds_including_validation_and_checkpoints'] for candidate in read(out / f'reranking_{kind}.json') for r in candidate['runs'])
        final_time = ga_runs[ga_runs.model == kind].training_seconds_including_validation_and_checkpoints.sum()
        total = search_time + rerank_time + final_time
        lines.append(f'| {kind} | {sum(r["status"]=="ok" for r in trials)} / {len(trials)} | {search_time:.1f} | {rerank_time:.1f} | {final_time:.1f} | {total:.1f} |')
        costs.append({'model': kind, 'search_seconds': search_time, 'rerank_seconds': rerank_time, 'final_training_seconds': final_time, 'total_seconds': total})
        for r in trials:
            search_rows.append({'model': kind, 'trial_id': r['trial_id'], 'generation': r['generation'], 'fitness': r['fitness'],
                                'seconds': r['elapsed_seconds'], 'status': r['status'], **r['genes']})
    pd.DataFrame(costs).to_csv(out / 'costs.csv', index=False)
    pd.DataFrame(search_rows).to_csv(out / 'search_trials.csv', index=False)
    lines += ['', 'Thời gian search gồm train, validation, checkpoint và dọn bộ nhớ giữa trial. Các cột còn lại gồm train, validation, checkpoint. Tổng không gồm tiền xử lý, test và sinh báo cáo. Cùng số trial không đồng nghĩa cùng thời gian GPU. Thời gian baseline được ghi từ lần chạy trước, không phải phép benchmark đồng thời.', '',
        '| Phương pháp | Tham số | Bộ nhớ tensor GPU đỉnh TB (MiB) | Train cuối TB / seed (giây) |', '|---|---:|---:|---:|']
    for method in order:
        subset = all_runs[all_runs.method == method]
        lines.append(f'| {method} | {int(subset.parameters.iloc[0])} | {subset.peak_cuda_allocated_mb.mean():.1f} | {subset.training_seconds_including_validation_and_checkpoints.mean():.1f} |')
    class_rows = []
    matrices = {}
    for method in order:
        is_ga = method.startswith('GA+')
        kind = method.removeprefix('GA+').lower()
        root = out / 'final' if is_ga else baseline
        scores = [read(root / f'{kind}_seed{seed}/metrics_test.json') for seed in spec['final']['seeds']]
        labels = scores[0]['confusion_matrix_labels']
        matrices[method] = np.mean([s['confusion_matrix_rows_true_columns_predicted'] for s in scores], axis=0)
        for label in labels:
            vals = [s['per_class'][label]['f1-score'] for s in scores]
            class_rows.append({'method': method, 'label': label, 'f1_mean': np.mean(vals), 'f1_std': np.std(vals, ddof=1)})
    per_class = pd.DataFrame(class_rows)
    per_class.to_csv(out / 'per_class_test.csv', index=False)
    pivot = per_class.pivot(index='label', columns='method', values='f1_mean').loc[labels, order]
    lines += ['', '## F1 test từng lớp', '', '| Nhãn | LSTM | GA+LSTM | xLSTM | GA+xLSTM |', '|---|---:|---:|---:|---:|']
    for label, row in pivot.iterrows():
        lines.append('| ' + label + ' | ' + ' | '.join(f'{v*100:.2f}' for v in row) + ' |')
    lines += ['', '## Giới hạn diễn giải', '',
        '- Test này đã được xem ở thực nghiệm cấu hình cố định trước. Đây là nghiên cứu mở rộng mang tính thăm dò trên cùng test, không phải một lần xác nhận trên test hoàn toàn mới.',
        '- Chỉ chạy một seed tìm kiếm GA cho mỗi kiến trúc. Ba seed cuối đo dao động huấn luyện của cấu hình đã chọn, không đo độ ổn định của chính quá trình tìm kiếm GA.',
        '- Giữ seed cuối giống baseline để so sánh theo cặp; seed 42 cũng được dùng ở vòng proxy. Do đó không coi cả ba seed cuối là hoàn toàn độc lập với quá trình tìm cấu hình.',
        '- Chưa có đối chứng random search cùng ngân sách, nên mức tăng so với cấu hình cố định không chứng minh GA tốt hơn các thuật toán tìm siêu tham số khác.',
        '- Fitness 5 epoch là đánh giá ngắn, có thể ưu tiên cấu hình học nhanh. Chạy lại top 2 giảm phần nào rủi ro nhưng không bảo đảm tìm được cấu hình tối ưu.',
        '- Dùng cùng không gian tìm kiếm/ngân sách epoch, nhưng số tham số, epoch dừng sớm và chi phí tính toán khác nhau. Không suy ra riêng tác động của từng thành phần kiến trúc.',
        '- Tất cả lựa chọn cấu hình và checkpoint dùng validation; việc lặp nhiều trial có thể làm quá khớp validation. Không điều chỉnh GA từ điểm test.',
        '- Cố định seed, bật deterministic ở chế độ warn_only; CUDA cumsum của mLSTM trong PyTorch 2.5.1 không bảo đảm tái lập từng bit.', '',
        '## Biểu đồ và tái lập', '', '![GA qua các trial và thời gian](search_progress.png)', '',
        '![So sánh kết quả](comparison_test.png)', '', '![F1 từng lớp](per_class_test.png)', '',
        '![Ma trận nhầm lẫn](confusion_test.png)', '',
        'Cấu hình: configs/ga_lstm_xlstm_comparison.json. Hướng dẫn: GA_EXPERIMENTS.md. Mỗi trial lưu checkpoint, lịch sử học, dự đoán validation và log; search_state.json lưu genealogy và quần thể mỗi thế hệ. locked_protocol.json và code/ lưu hash cùng bản mã nguồn chạy.', '',
        '```powershell', 'python scripts/check_ga_search.py', 'python scripts/run_ga_comparison.py --outdir results/new_ga_run',
        'python scripts/report_ga_comparison.py --outdir results/new_ga_run', 'python scripts/verify_ga_comparison.py --outdir results/new_ga_run', '```', '',
        'Nguồn kiến trúc xLSTM: [NX-AI/xlstm](https://github.com/NX-AI/xlstm); dữ liệu: [UIT-VSMEC](https://arxiv.org/abs/1911.09339).', '']
    (out / 'REPORT_VI.md').write_text('\n'.join(lines), encoding='utf-8')
    plt.rcParams.update({'figure.dpi': 140, 'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for kind in spec['models']:
        t = pd.DataFrame([r for r in search_rows if r['model'] == kind])
        best = t.fitness.fillna(-np.inf).cummax() * 100
        axes[0].plot(t.trial_id + 1, best, marker='.', label=kind)
        axes[1].plot(t.seconds.cumsum(), best, marker='.', label=kind)
    for ax, xlabel in zip(axes, ['Unique trial number', 'Cumulative search time (seconds)']):
        ax.set(xlabel=xlabel, ylabel='Best proxy validation Macro-F1 (%)')
        ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out / 'search_progress.png'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    mean = summary[('test_macro_f1', 'mean')] * 100
    std = summary[('test_macro_f1', 'std')] * 100
    ax.bar(order, mean, yerr=std, capsize=5, color=['#7fafd4', '#245d8b', '#e2ae74', '#b26d25'])
    for i, value in enumerate(mean):
        ax.text(i, value + std.iloc[i] + 1, f'{value:.2f}', ha='center')
    ax.set(ylabel='Test Macro-F1 (%)', ylim=(0, min(100, max(mean + std) + 8)), title='Mean +/- sample SD over three training seeds')
    fig.tight_layout(); fig.savefig(out / 'comparison_test.png'); plt.close(fig)
    ax = (pivot * 100).plot.bar(figsize=(12, 4), rot=0, ylabel='Test F1 (%)')
    ax.figure.tight_layout(); ax.figure.savefig(out / 'per_class_test.png'); plt.close(ax.figure)
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    for ax, method in zip(axes.flat, order):
        cm = matrices[method]
        cm = cm / cm.sum(axis=1, keepdims=True) * 100
        ax.imshow(cm, vmin=0, vmax=100, cmap='Blues')
        ax.set(xticks=range(7), yticks=range(7), xticklabels=labels, yticklabels=labels,
               title=f'{method}: mean test confusion (%)', xlabel='Predicted', ylabel='True')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        for i in range(7):
            for j in range(7):
                ax.text(j, i, f'{cm[i,j]:.0f}', ha='center', va='center', color='white' if cm[i,j] > 50 else 'black')
    fig.tight_layout(); fig.savefig(out / 'confusion_test.png'); plt.close(fig)
    print(out / 'REPORT_VI.md')


if __name__ == '__main__':
    main()
