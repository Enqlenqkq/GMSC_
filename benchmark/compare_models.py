"""저장된 benchmark와 과제 모델의 입력 구조 확인 및 공통 Test 비교."""
from pathlib import Path
import argparse
import json
import sys
import hashlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def describe(obj):
    info = {'type': f'{type(obj).__module__}.{type(obj).__name__}'}
    for name in ['n_features_in_', 'feature_names_in_', 'classes_', 'n_samples_seen_']:
        if hasattr(obj, name):
            value = getattr(obj, name)
            info[name] = value.tolist() if hasattr(value, 'tolist') else value
    if isinstance(obj, dict):
        info['entries'] = {str(k): describe(v) if hasattr(v, 'predict') or isinstance(v, dict)
                           else {'type': type(v).__name__, 'value': str(v)[:500]}
                           for k, v in obj.items()}
    if hasattr(obj, 'steps'):
        info['steps'] = {k: describe(v) for k, v in obj.steps}
    if hasattr(obj, 'get_params'):
        params = obj.get_params(deep=False)
        info['params'] = {k: v for k, v in params.items()
                          if v is None or isinstance(v, (str, int, float, bool))}
    return info


def compare_models():
    """모델별 입력 스키마를 유지한 공통 Test 기술 비교. 학습·튜닝은 하지 않는다."""
    import joblib
    import numpy as np
    import pandas as pd
    import matplotlib
    if __name__ == '__main__':
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        average_precision_score, roc_auc_score, precision_score, recall_score,
        f1_score, fbeta_score, matthews_corrcoef, brier_score_loss,
        confusion_matrix, PrecisionRecallDisplay, RocCurveDisplay,
        ConfusionMatrixDisplay,
    )

    out = ROOT / 'submission/benchmark_comparison'
    out.mkdir(parents=True, exist_ok=True)
    benchmark = joblib.load(ROOT / 'benchmark/finalized_credit_scoring_model.joblib')
    ours = joblib.load(ROOT / 'submission/final_model.joblib')
    data = pd.read_csv(ROOT / 'givemesomecredit.csv')
    split = np.load(ROOT / 'split_indices.npz')
    assert len(set(split['train_idx']) & set(split['test_idx'])) == 0
    raw_test = data.loc[split['test_idx']].copy()
    y = raw_test['SeriousDlqin2yrs']
    assert set(y.unique()) == {0, 1}

    # benchmark는 외부 전처리기가 없는 단일 모델이다. 저장된 원본 컬럼명을 따른다.
    # 별도 로그·스케일 변환이 학습 전에 적용됐다면 추가 정보에 맞춰 수정해야 한다.
    bench_names = list(benchmark.feature_names_in_)
    X_bench = raw_test[bench_names]
    renamed = raw_test.rename(columns={
        'RevolvingUtilizationOfUnsecuredLines': 'RevUtil', 'age': 'Age',
        'NumberOfTime30-59DaysPastDueNotWorse': 'PastDue_30-59',
        'MonthlyIncome': 'Income', 'NumberOfDependents': 'Dependents',
        'CombinedDefaulted': 'CombDefault', 'CombinedCreditLoans': 'CombCredit',
    })
    X_ours = renamed[ours['input_columns']]
    benchmark_positive = list(benchmark.classes_).index(1)
    ours_positive = list(ours['pipeline'].classes_).index(ours['positive_label'])
    probabilities = {
        'Benchmark XGB': benchmark.predict_proba(X_bench)[:, benchmark_positive],
        'Final XGB': ours['pipeline'].predict_proba(X_ours)[:, ours_positive],
    }
    for prob in probabilities.values():
        assert len(prob) == len(y) and np.isfinite(prob).all()
        assert ((prob >= 0) & (prob <= 1)).all()

    # 현재 모델의 저장된 최초 평가 확률과 재로드 결과가 일치하는지 확인한다.
    frozen = pd.read_csv(ROOT / 'final_test_predictions.csv')
    assert np.array_equal(frozen['row_index'].to_numpy(), raw_test.index.to_numpy())
    assert np.array_equal(frozen['y_true'].to_numpy(), y.to_numpy())
    np.testing.assert_allclose(probabilities['Final XGB'], frozen['prob'], rtol=0, atol=1e-6)

    rows = []
    conditions = [('Benchmark XGB', 0.5, 'Common threshold 0.5'),
                  ('Final XGB', 0.5, 'Common threshold 0.5'),
                  ('Final XGB', float(ours['threshold']), 'Train OOF F2 threshold')]
    for name, threshold, rule in conditions:
        prob = probabilities[name]
        pred = (prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        rows.append({
            'model': name, 'threshold_rule': rule, 'threshold': threshold,
            'AP': average_precision_score(y, prob), 'AUROC': roc_auc_score(y, prob),
            'Precision': precision_score(y, pred, zero_division=0),
            'Recall': recall_score(y, pred, zero_division=0),
            'F1': f1_score(y, pred, zero_division=0),
            'F2': fbeta_score(y, pred, beta=2, zero_division=0),
            'MCC': matthews_corrcoef(y, pred), 'Brier': brier_score_loss(y, prob),
            'TP': int(tp), 'FN': int(fn), 'FP': int(fp), 'TN': int(tn),
        })
    results = pd.DataFrame(rows)
    results.to_csv(out / 'metrics.csv', index=False)
    pd.DataFrame({
        'row_index': raw_test.index.to_numpy(), 'y_true': y.to_numpy(),
        'benchmark_prob': probabilities['Benchmark XGB'],
        'final_prob': probabilities['Final XGB'],
    }).to_csv(out / 'predictions.csv', index=False)

    colors = ['#193449', '#199C9C']
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for (name, prob), color in zip(probabilities.items(), colors):
        PrecisionRecallDisplay.from_predictions(y, prob, name=name, ax=axes[0, 0], curve_kwargs={'color': color})
        RocCurveDisplay.from_predictions(y, prob, name=name, ax=axes[0, 1], curve_kwargs={'color': color})
    axes[0, 0].axhline(y.mean(), color='gray', linestyle='--', label='Positive prevalence')
    axes[0, 0].set_title('Precision–Recall: same current Test')
    axes[0, 0].legend(loc='upper right')
    axes[0, 1].plot([0, 1], [0, 1], '--', color='gray')
    axes[0, 1].set_title('ROC: same current Test')
    common = results.iloc[:2].set_index('model')
    for ax, metrics, title in [
        (axes[1, 0], ['AP', 'AUROC'], 'Ranking metrics'),
        (axes[1, 1], ['Precision', 'Recall', 'F2', 'MCC'], 'Common threshold = 0.5'),
    ]:
        common[metrics].T.plot(kind='bar', ax=ax, rot=0, color=colors)
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Score')
        ax.legend(loc='upper left', fontsize=9)
        for bar in ax.patches:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.012,
                    f'{bar.get_height():.3f}', ha='center', fontsize=8)
    fig.suptitle('Reference comparison: benchmark training/Test overlap is unknown', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out / 'model_comparison.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, row in zip(axes, rows):
        pred = (probabilities[row['model']] >= row['threshold']).astype(int)
        ConfusionMatrixDisplay.from_predictions(y, pred, labels=[0, 1],
            ax=ax, cmap='Blues', colorbar=False, values_format='d')
        ax.set_title(f"{row['model']}\nthreshold={row['threshold']:.4f}")
    fig.suptitle('Shared Test; final F2 threshold was selected on our Train OOF', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out / 'confusion_matrices.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    final_row, bench_row = rows[1], rows[0]
    notes = f'''Benchmark 참고 비교

평가 데이터: 현재 과제의 고정 Test {len(y):,}건, 양성 {int(y.sum()):,}건.
사용자 확인: benchmark는 다른 데이터 또는 다른 분할로 학습했다.
benchmark 학습 표본과 현재 Test의 중복 여부는 확인되지 않아 독립적인 우열 검증이 아니다.
benchmark는 원본 컬럼 8개를 받는 XGB 단일 모델이다. 별도 전처리 정보가 제공되지 않아
저장된 컬럼 순서로 원본 수치를 입력했다. 외부 전처리가 있었다면 결과를 재해석해야 한다.
현재 모델은 저장된 전처리·스케일러·XGB 파이프라인을 그대로 적용했다.
두 모델 모두 추가 학습이나 threshold 튜닝을 하지 않았다.

공통 threshold=0.5
- Benchmark: AP={bench_row['AP']:.6f}, AUROC={bench_row['AUROC']:.6f}, Recall={bench_row['Recall']:.6f}, Precision={bench_row['Precision']:.6f}.
- Final: AP={final_row['AP']:.6f}, AUROC={final_row['AUROC']:.6f}, Recall={final_row['Recall']:.6f}, Precision={final_row['Precision']:.6f}.
- Final - Benchmark: AP={final_row['AP']-bench_row['AP']:+.6f}, AUROC={final_row['AUROC']-bench_row['AUROC']:+.6f}.

최종 모델의 Train OOF F2 threshold={ours['threshold']:.6f} 결과는 별도 행에 표시했다.
benchmark의 최적 threshold는 제공되지 않았으므로 이 행과 benchmark@0.5를 비교해
모델 자체의 개선이라고 해석하지 않는다. 이 결과로 최종 모델을 다시 선정하지 않는다.
'''
    (out / '비교해석.txt').write_text(notes, encoding='utf-8')

    def sanitize(obj):
        if isinstance(obj, dict):
            return {str(k): sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        if isinstance(obj, float) and not np.isfinite(obj):
            return str(obj)
        return obj
    metadata = {
        'evaluation': 'Reference only; benchmark training overlap unknown',
        'benchmark_input_assumption': 'Raw original-named numeric columns; no external preprocessing supplied',
        'benchmark': describe(benchmark), 'ours': describe(ours),
        'benchmark_feature_types': benchmark.get_booster().feature_types,
        'test_rows': len(y), 'test_positive': int(y.sum()),
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [ROOT/'benchmark/finalized_credit_scoring_model.joblib',
                                    ROOT/'submission/final_model.joblib', ROOT/'split_indices.npz',
                                    ROOT/'givemesomecredit.csv']},
    }
    (out / 'metadata.json').write_text(json.dumps(sanitize(metadata), ensure_ascii=False,
                                                 indent=2, allow_nan=False), encoding='utf-8')
    print(results.to_string(index=False), flush=True)
    print(f'시각화 저장: {out}', flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspect', action='store_true')
    args = parser.parse_args()
    if args.inspect:
        import joblib
        obj = joblib.load(ROOT/'benchmark/finalized_credit_scoring_model.joblib')
        info = describe(obj)
        info['booster_feature_types'] = obj.get_booster().feature_types
        print(json.dumps(info, ensure_ascii=False, indent=2, default=str))
    else:
        compare_models()


if __name__ == '__main__':
    main()
