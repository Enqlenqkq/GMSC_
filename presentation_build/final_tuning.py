"""마지막 Train 전용 탐색과 제출용 근거 집계. 기존 Test 결과를 덮어쓰지 않는다."""
from pathlib import Path
import ast
import json
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, ParameterSampler, GridSearchCV, cross_val_predict
from sklearn.metrics import precision_recall_curve, fbeta_score, precision_score, recall_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'submission'
OUT.mkdir(exist_ok=True)
nb = json.loads((ROOT / 'ass.ipynb').read_text(encoding='utf-8'))
class_source = next(''.join(c['source']) for c in nb['cells']
                    if c['cell_type'] == 'code' and 'class GMSCPreprocessor(' in ''.join(c['source']))
tree = ast.parse(class_source)
class_node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GMSCPreprocessor')
exec(compile(ast.Module(body=[class_node], type_ignores=[]), '<notebook preprocessor>', 'exec'))

data = pd.read_csv(ROOT / 'givemesomecredit.csv').rename(columns={
    'SeriousDlqin2yrs': 'Dlq2yrs',
    'RevolvingUtilizationOfUnsecuredLines': 'RevUtil', 'age': 'Age',
    'NumberOfTime30-59DaysPastDueNotWorse': 'PastDue_30-59',
    'MonthlyIncome': 'Income', 'NumberOfDependents': 'Dependents',
    'CombinedDefaulted': 'CombDefault', 'CombinedCreditLoans': 'CombCredit',
})
split = np.load(ROOT / 'split_indices.npz')
train_idx, test_idx = split['train_idx'], split['test_idx']
assert len(set(train_idx) & set(test_idx)) == 0
X_train = data.loc[train_idx].drop(columns='Dlq2yrs')
y_train = data.loc[train_idx, 'Dlq2yrs']
assert X_train.shape == (120000, 8)
cv_splits = list(StratifiedKFold(5, shuffle=True, random_state=33).split(X_train, y_train))

pipe = Pipeline([
    ('clean', GMSCPreprocessor(use_log=True)),
    ('scale', StandardScaler()),
    ('model', XGBClassifier(tree_method='hist', eval_metric='logloss',
                            scale_pos_weight=1.0, random_state=33, n_jobs=1)),
])
space = {
    'model__n_estimators': [250, 500, 800],
    'model__max_depth': [2, 3, 4],
    'model__learning_rate': [0.03, 0.05, 0.1],
    'model__min_child_weight': [1, 5, 10],
    'model__subsample': [0.8, 1.0],
    'model__reg_lambda': [1.0, 5.0, 10.0],
}
incumbent = {
    'model__n_estimators': 250, 'model__max_depth': 3,
    'model__learning_rate': 0.05, 'model__min_child_weight': 1,
    'model__subsample': 1.0, 'model__reg_lambda': 1.0,
}
candidates = [incumbent]
for candidate in ParameterSampler(space, n_iter=30, random_state=33):
    if candidate not in candidates:
        candidates.append(candidate)
    if len(candidates) == 20:
        break
grid = [{k: [v] for k, v in c.items()} for c in candidates]
search = GridSearchCV(pipe, grid, scoring={'AP': 'average_precision', 'AUROC': 'roc_auc'},
                      refit='AP', cv=cv_splits, n_jobs=3, pre_dispatch=3,
                      error_score='raise', verbose=1)
search.fit(X_train, y_train)
results = pd.DataFrame(search.cv_results_).sort_values('rank_test_AP')
results.to_csv(OUT / 'followup_tuning_results.csv', index=False)
best = results.iloc[0]
previous = pd.read_csv(ROOT / 'xgb_tuning_results.csv').sort_values('rank_test_AP').iloc[0]

# 후속 모델의 OOF도 Train 전용 개발 평가다. Test 성적은 갱신하지 않는다.
prob = cross_val_predict(clone(search.best_estimator_), X_train, y_train,
                        method='predict_proba', cv=cv_splits, n_jobs=3, pre_dispatch=3)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_train, prob)
den = 4 * precision[:-1] + recall[:-1]
f2 = np.divide(5 * precision[:-1] * recall[:-1], den, out=np.zeros_like(den), where=den > 0)
threshold = float(thresholds[int(np.argmax(f2))])
pred = (prob >= threshold).astype(int)
pd.DataFrame({'row_index': X_train.index, 'y_true': y_train.to_numpy(), 'prob': prob}).to_csv(
    OUT / 'followup_train_oof.csv', index=False)

def native(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)

report = {
    'followup': {
        'status': 'completed_train_only', 'n_candidates': len(candidates),
        'cv_AP': float(best['mean_test_AP']), 'cv_AUROC': float(best['mean_test_AUROC']),
        'previous_cv_AP': float(previous['mean_test_AP']),
        'AP_delta': float(best['mean_test_AP'] - previous['mean_test_AP']),
        'best_params': search.best_params_, 'oof_threshold': threshold,
        'oof_F2': float(fbeta_score(y_train, pred, beta=2)),
        'oof_precision': float(precision_score(y_train, pred)),
        'oof_recall': float(recall_score(y_train, pred)),
        'test_evaluated': False,
        'limitation': 'Test 확인 후 추가 탐색이며 Train CV만 사용. 기존 Test 성적은 최초 고정 모델의 평가 기록.',
    },
    'baseline': pd.read_csv(ROOT / 'baseline_summary.csv').to_dict('records'),
    'feature': pd.read_csv(ROOT / 'feature_summary.csv').to_dict('records'),
    'balance': pd.read_csv(ROOT / 'balance_summary.csv', keep_default_na=False).to_dict('records'),
    'frozen_test': pd.read_csv(ROOT / 'final_test_metrics.csv').to_dict('records'),
    'frozen_config': json.loads((ROOT / 'final_config.json').read_text(encoding='utf-8')),
    'data': {'rows': len(data), 'train': len(train_idx), 'test': len(test_idx),
             'positive': int(data['Dlq2yrs'].sum()),
             'negative': int((data['Dlq2yrs'] == 0).sum()),
             'raw_features': 8, 'model_features': 9},
}
(OUT / 'presentation_evidence.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, default=native), encoding='utf-8')
print(json.dumps(report['followup'], ensure_ascii=False, default=native), flush=True)
