"""benchmark 노트북의 seed=7 분할을 재구성하고 비교의 학습 중복을 점검한다."""
from pathlib import Path
import sys
import json
import csv
import hashlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import joblib
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.base import clone
    from sklearn.metrics import (average_precision_score, roc_auc_score, precision_score,
                                 recall_score, fbeta_score, matthews_corrcoef,
                                 brier_score_loss, confusion_matrix,
                                 PrecisionRecallDisplay, RocCurveDisplay)

    out = ROOT / 'submission/benchmark_comparison'
    out.mkdir(exist_ok=True, parents=True)
    data = pd.read_csv(ROOT/'givemesomecredit.csv')
    y = data['SeriousDlqin2yrs']
    bench = joblib.load(ROOT/'benchmark/finalized_credit_scoring_model.joblib')
    ours = joblib.load(ROOT/'submission/final_model.joblib')
    ours_split = np.load(ROOT/'split_indices.npz')
    train7, test7 = train_test_split(np.arange(len(data)), test_size=0.2,
                                    stratify=y, random_state=7)
    test33 = ours_split['test_idx']
    common = np.intersect1d(test7, test33)
    seen = np.intersect1d(train7, test33)
    columns = list(bench.feature_names_in_)
    X = data[columns]
    renamed = data.rename(columns={
        'RevolvingUtilizationOfUnsecuredLines': 'RevUtil', 'age': 'Age',
        'NumberOfTime30-59DaysPastDueNotWorse': 'PastDue_30-59',
        'MonthlyIncome': 'Income', 'NumberOfDependents': 'Dependents',
        'CombinedDefaulted': 'CombDefault', 'CombinedCreditLoans': 'CombCredit',
    })[ours['input_columns']]

    # 현재 CSV와 노트북 최종 설정·seed=7로 재학습해 저장 모델과 비교한다.
    recreated = clone(bench)
    recreated.fit(X.iloc[train7], y.iloc[train7])
    saved_prob = bench.predict_proba(X)[:, list(bench.classes_).index(1)]
    reconstructed_prob = recreated.predict_proba(X)[:, list(recreated.classes_).index(1)]
    max_abs = float(np.max(np.abs(saved_prob-reconstructed_prob)))
    same_trees = bench.get_booster().get_dump(dump_format='json') == recreated.get_booster().get_dump(dump_format='json')
    exact_prob = bool(np.array_equal(saved_prob, reconstructed_prob))
    print(f'Reconstruction: max_abs_probability_difference={max_abs}, identical_trees={same_trees}', flush=True)

    our_prob = ours['pipeline'].predict_proba(renamed)[:, list(ours['pipeline'].classes_).index(1)]
    all_rows = []
    for cohort, indices in [('Current Test (seed 33)', test33),
                            ('Common Test intersection (7 & 33)', common)]:
        truth = y.iloc[indices]
        for name, prob_all, threshold in [
            ('Benchmark XGB', saved_prob, 0.5),
            ('Final XGB', our_prob, 0.5),
            ('Final XGB (OOF threshold)', our_prob, float(ours['threshold'])),
        ]:
            prob = prob_all[indices]
            pred = (prob >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(truth, pred, labels=[0, 1]).ravel()
            all_rows.append({
                'cohort': cohort, 'n': len(indices), 'positive': int(truth.sum()),
                'model': name, 'threshold': threshold,
                'AP': average_precision_score(truth, prob), 'AUROC': roc_auc_score(truth, prob),
                'Precision': precision_score(truth, pred, zero_division=0),
                'Recall': recall_score(truth, pred, zero_division=0),
                'F2': fbeta_score(truth, pred, beta=2, zero_division=0),
                'MCC': matthews_corrcoef(truth, pred), 'Brier': brier_score_loss(truth, prob),
                'TP': int(tp), 'FN': int(fn), 'FP': int(fp), 'TN': int(tn),
            })
    results = pd.DataFrame(all_rows)
    results.to_csv(out/'split_audit_metrics.csv', index=False)
    pd.DataFrame({'row_index': common}).to_csv(out/'common_test_indices.csv', index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, prob, color in [('Benchmark XGB', saved_prob, '#193449'),
                               ('Final XGB', our_prob, '#199C9C')]:
        PrecisionRecallDisplay.from_predictions(y.iloc[common], prob[common], name=name,
            ax=axes[0], curve_kwargs={'color': color})
        RocCurveDisplay.from_predictions(y.iloc[common], prob[common], name=name,
            ax=axes[1], curve_kwargs={'color': color})
    axes[0].axhline(y.iloc[common].mean(), linestyle='--', color='gray')
    axes[0].set_title('PR: reconstructed common Test intersection')
    axes[0].legend(loc='upper right')
    axes[1].set_title('ROC: reconstructed common Test intersection')
    axes[1].plot([0,1], [0,1], '--', color='gray')
    fig.suptitle(f'Shared held-out rows under reconstructed splits: n={len(common):,}', fontsize=13)
    fig.tight_layout(rect=[0,0,1,0.94])
    fig.savefig(out/'common_test_curves.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cohort_names = ['Current Test (seed 33)', 'Common Test intersection (7 & 33)']
    for ax, metric in zip(axes, ['AP', 'AUROC']):
        table = results[results['threshold']==0.5].pivot(index='cohort', columns='model', values=metric).reindex(cohort_names)
        table.plot(kind='bar', ax=ax, color=['#193449','#199C9C'], rot=0)
        ax.set_title(metric)
        ax.set_ylim(0,1)
        ax.set_xticklabels(['Current Test\ncontains benchmark training rows', 'Common Test\nreconstructed intersection'], fontsize=9)
        for bar in ax.patches:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.012,
                    f'{bar.get_height():.4f}', ha='center', fontsize=9)
        ax.set_xlabel('')
        ax.legend_.remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,0.96), ncol=2, frameon=False)
    fig.suptitle('How split overlap affects the interpretation of model comparison', fontsize=13)
    fig.tight_layout(rect=[0,0,1,0.88])
    fig.savefig(out/'split_overlap_comparison.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    report = {
        'assumption': 'Notebook seed=7 split reconstructed on current CSV with original row order.',
        'notebook_source': 'benchmark/creditanalysis.ipynb',
        'reconstruction_identical_probabilities_on_all_rows': exact_prob,
        'reconstruction_identical_tree_dumps': same_trees,
        'reconstruction_max_abs_probability_difference': max_abs,
        'benchmark_original_test_AUROC_recomputed': float(roc_auc_score(y.iloc[test7], saved_prob[test7])),
        'benchmark_notebook_reported_test_AUROC': 0.8577,
        'current_test_n': len(test33),
        'current_test_in_reconstructed_benchmark_train': len(seen),
        'common_test_n': len(common),
        'common_test_positive': int(y.iloc[common].sum()),
        'limitation': '추가 사후 분석. 공통 Test도 이미 개발·해석에 반복 사용되어 새로운 외부 검증은 아니다. 원본 학습 파일은 제공되지 않았으나 현재 CSV 재학습과 저장 모델의 일치 여부를 함께 기록한다.',
        'metrics': all_rows,
    }
    (out/'notebook_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='metrics'}, ensure_ascii=False, indent=2), flush=True)
    print(results.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
