"""저장된 최종 XGB에 대한 Train 배경 기반 TreeSHAP 분석. 모델은 재학습하지 않는다."""
from pathlib import Path
import json
import hashlib

ROOT = Path(__file__).resolve().parent


def run_xai():
    import joblib
    import numpy as np
    import pandas as pd
    import shap
    import matplotlib
    if __name__ == '__main__':
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split

    out = ROOT/'submission/xai'
    out.mkdir(parents=True, exist_ok=True)
    model_path = ROOT/'submission/final_model.joblib'
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    artifact = joblib.load(model_path)
    pipeline = artifact['pipeline']
    model = pipeline.named_steps['model']
    seed = int(artifact['config']['seed'])
    threshold = float(artifact['threshold'])
    data = pd.read_csv(ROOT/'givemesomecredit.csv').rename(columns={
        'SeriousDlqin2yrs': 'Dlq2yrs', 'RevolvingUtilizationOfUnsecuredLines': 'RevUtil',
        'age': 'Age', 'NumberOfTime30-59DaysPastDueNotWorse': 'PastDue_30-59',
        'MonthlyIncome': 'Income', 'NumberOfDependents': 'Dependents',
        'CombinedDefaulted': 'CombDefault', 'CombinedCreditLoans': 'CombCredit',
    })
    split = np.load(ROOT/'split_indices.npz')
    X_train = data.loc[split['train_idx'], artifact['input_columns']]
    X_test = data.loc[split['test_idx'], artifact['input_columns']]
    y_test = data.loc[split['test_idx'], 'Dlq2yrs']

    # 학습된 변환만 적용한다. 배경은 Train 실제 표본, 설명 대상은 Test 층화 표본이다.
    background_rows = X_train.sample(n=100, random_state=seed)
    explain_positions, _ = train_test_split(np.arange(len(X_test)), train_size=1000,
                                            stratify=y_test, random_state=seed)
    explain_rows = X_test.iloc[explain_positions]
    truth = y_test.iloc[explain_positions].to_numpy()
    names = list(pipeline.named_steps['scale'].feature_names_in_)
    background = pd.DataFrame(pipeline[:-1].transform(background_rows),
                              columns=names, index=background_rows.index)
    transformed = pd.DataFrame(pipeline[:-1].transform(explain_rows),
                               columns=names, index=explain_rows.index)

    explainer = shap.TreeExplainer(model, data=background, model_output='raw',
                                  feature_perturbation='interventional')
    explanation = explainer(transformed, check_additivity=True)
    values = np.asarray(explanation.values)
    base = np.asarray(explanation.base_values)
    assert values.shape == (1000, 9)
    margin = model.predict(transformed.to_numpy(), output_margin=True)
    reconstructed_margin = base + values.sum(axis=1)
    np.testing.assert_allclose(reconstructed_margin, margin, rtol=1e-5, atol=1e-5)
    prob = pipeline.predict_proba(explain_rows)[:, list(pipeline.classes_).index(1)]
    np.testing.assert_allclose(1/(1+np.exp(-reconstructed_margin)), prob, rtol=1e-5, atol=1e-6)
    pred = (prob >= threshold).astype(int)

    # 표시되는 값도 실제 모델 입력인 로그·z-score 변환값이다.
    labels = [f'{n} (log1p, z)' if n in ['RevUtil', 'DebtRatio', 'Income'] else f'{n} (z)' for n in names]
    display_exp = shap.Explanation(values=values, base_values=base,
                                    data=transformed.to_numpy(), feature_names=labels)
    importance = pd.DataFrame({'feature': names, 'mean_abs_shap': np.abs(values).mean(axis=0)})
    importance = importance.sort_values('mean_abs_shap', ascending=False)
    importance.to_csv(out/'global_importance.csv', index=False)
    details = pd.DataFrame(values, columns=names, index=explain_rows.index)
    details.insert(0, 'row_index', explain_rows.index)
    details.insert(1, 'y_true', truth)
    details.insert(2, 'probability', prob)
    details.insert(3, 'prediction', pred)
    details.insert(4, 'base_log_odds', base)
    details.to_csv(out/'shap_values.csv', index=False)

    shap.plots.bar(display_exp, max_display=9, show=False)
    fig = plt.gcf()
    fig.set_size_inches(11, 6)
    plt.title('Final XGB: mean |SHAP| (log-odds)', pad=20)
    fig.savefig(out/'global_importance.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    shap.plots.beeswarm(display_exp, max_display=9, show=False, plot_size=(11, 6))
    plt.title('Final XGB: SHAP direction (transformed feature values)', pad=20)
    plt.gcf().savefig(out/'beeswarm.png', dpi=200, bbox_inches='tight')
    plt.close(plt.gcf())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, feature in zip(axes, importance['feature'].head(2)):
        j = names.index(feature)
        ax.scatter(transformed[feature], values[:, j], s=10, alpha=0.4, color='#199C9C')
        ax.axhline(0, color='gray', linewidth=1, linestyle='--')
        ax.set_xlabel(labels[j])
        ax.set_ylabel('SHAP value (log-odds)')
        ax.set_title(feature)
    fig.tight_layout()
    fig.savefig(out/'top_feature_dependence.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    # 극단 예측과 오분류 사례를 의도적으로 선택한다. 대표적인 고객이라는 뜻은 아니다.
    cases = [('high_risk', int(np.argmax(prob))), ('low_risk', int(np.argmin(prob)))]
    used = {idx for _, idx in cases}
    for label, mask, descending in [('false_positive', (pred==1)&(truth==0), True),
                                    ('false_negative', (pred==0)&(truth==1), False)]:
        indices = np.flatnonzero(mask)
        indices = indices[np.argsort(prob[indices])]
        if descending:
            indices = indices[::-1]
        chosen = next((int(i) for i in indices if i not in used), None)
        if chosen is not None:
            cases.append((label, chosen))
            used.add(chosen)
    case_rows, contribution_rows = [], []
    for label, i in cases:
        shap.plots.waterfall(display_exp[i], max_display=9, show=False)
        fig = plt.gcf()
        fig.set_size_inches(11, 6)
        fig.suptitle(f'{label}: row {explain_rows.index[i]} | P(1)={prob[i]:.3f} | true={truth[i]} | predicted={pred[i]}', fontsize=12, y=1.03)
        fig.savefig(out/f'waterfall_{label}.png', dpi=200, bbox_inches='tight')
        plt.close(fig)
        row = {'case': label, 'row_index': int(explain_rows.index[i]),
               'y_true': int(truth[i]), 'prediction': int(pred[i]),
               'probability': float(prob[i]), 'threshold': threshold,
               'base_log_odds': float(base[i]), 'output_log_odds': float(margin[i])}
        row.update({f'raw_{col}': float(explain_rows.iloc[i][col]) for col in artifact['input_columns']})
        case_rows.append(row)
        for j, name in enumerate(names):
            contribution_rows.append({'case': label, 'feature': name,
                                      'model_input': float(transformed.iloc[i,j]),
                                      'shap_log_odds': float(values[i,j])})
    pd.DataFrame(case_rows).to_csv(out/'local_cases.csv', index=False)
    pd.DataFrame(contribution_rows).to_csv(out/'local_contributions.csv', index=False)

    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == model_hash
    report = {
        'model_sha256': model_hash, 'shap_version': shap.__version__,
        'explainer': 'TreeExplainer', 'feature_perturbation': 'interventional',
        'model_output': 'raw log-odds for binary XGB',
        'background': '100 real Train rows, simple random sample',
        'explanation': '1000 Test rows, stratified sample', 'seed': seed,
        'n_background': len(background), 'n_explained': len(explain_rows),
        'positive_in_sample': int(truth.sum()), 'threshold': threshold,
        'background_row_indices': [int(v) for v in background_rows.index],
        'explained_row_indices': [int(v) for v in explain_rows.index],
        'max_additivity_error': float(np.max(np.abs(margin-reconstructed_margin))),
        'importance': importance.to_dict('records'), 'cases': case_rows,
        'limitations': [
            'SHAP 설명은 인과 효과나 실제 연체 원인의 증명이 아니다.',
            '전역 중요도는 층화 Test 표본 1000건의 평균 절대 기여도다.',
            'SHAP 단위는 확률/퍼센트가 아닌 log-odds다. 합계를 sigmoid로 변환하면 모델 확률이 된다.',
            '표시 특성 값은 학습된 로그·스케일 변환값이다. 원 단위 값은 local_cases.csv에 별도 기록한다.',
            '상관된 특성의 의존성을 단순화하는 interventional 가정과 배경 표본에 따라 기여도가 달라질 수 있다.',
            '극단 예측·오분류 사례는 설명용으로 선택됐으며 전체 모집단의 대표 사례가 아니다.',
            '이 설명으로 Test 결과를 보고 모델을 재선택하거나 파라미터를 변경하지 않았다.',
        ],
    }
    (out/'xai_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    top = ', '.join(importance['feature'].head(3))
    notes = f'''최종 XGB SHAP 분석

전역 중요도 상위3: {top}
Train 실제 표본100건을 배경으로 Test 층화표본1000건(양성{truth.sum()}건)을 설명했다.
최종 모델은 재학습하거나 수정하지 않았다. 모델 파일 SHA-256={model_hash}
SHAP 합계+기준값과 모델 raw margin 일치 확인: 최대 오차 {report['max_additivity_error']:.8f}

그래프 해석
- global_importance.png: 평균 |SHAP|가 클수록 해당 표본에서 모델 예측에 큰 기여를 한다.
- beeswarm.png: SHAP 양수는 양성 점수를 높이고 음수는 낮춘다. 빨강은 변환된 특성값이 높고 파랑은 낮다.
- top_feature_dependence.png: 상위 특성의 모델 입력값과 SHAP 기여도 관계를 보여준다.
- waterfall_*.png: 기준 log-odds에 각 기여도를 더해 해당 행의 모델 점수에 도달한다.
- 원 단위 입력과 실제 라벨·예측 확률·선택 임계값은 local_cases.csv에서 확인한다.

주의
모든 SHAP 수치는 확률이나 퍼센트가 아니라 log-odds 단위다.
0보다 큰 기여는 실제 부도의 인과 원인이라는 뜻이 아니다.
극단값 사례는 설명 목적의 선택이며 모델 전체 성능이나 일반적 고객을 대표하지 않는다.
배경 표본 및 상관 특성 처리 가정에 따라 중요도·기여도가 달라질 수 있다.
설명 값만을 근거로 개인에 대한 자동화된 금융 의사결정의 정당성을 주장하지 않는다.

근거: https://shap.readthedocs.io/en/stable/generated/shap.TreeExplainer.html
'''
    (out/'XAI_해석안내.txt').write_text(notes, encoding='utf-8')
    print(importance.to_string(index=False), flush=True)
    print(pd.DataFrame(case_rows)[['case','row_index','probability','y_true','prediction']].to_string(index=False), flush=True)
    print(f'XAI 저장 완료: {out}; additivity error={report["max_additivity_error"]:.8f}', flush=True)
    return report


if __name__ == '__main__':
    run_xai()
