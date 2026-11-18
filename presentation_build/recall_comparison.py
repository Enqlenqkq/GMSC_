"""노트북에 Recall 비교 셀을 추가하고, --run 지정 시 기존 환경에서 실행한다."""
import ast
import json
from pathlib import Path
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / 'ass.ipynb'
nb = json.loads(path.read_text(encoding='utf-8'))
heading = '### 11. 기준 6개 모델과 튜닝 XGB의 Test Recall 비교'
source = '''# 같은 Test에서 기준 모델과 최종 모델을 기술적으로 비교한다.
# 이 표를 보고 모델을 다시 선택하거나 threshold를 변경하지 않는다.
from pathlib import Path
import json
from sklearn.metrics import confusion_matrix

recall_out = Path("submission")
recall_out.mkdir(exist_ok=True)
recall_rows = []

def recall_metrics(label, model_name, prob, threshold, condition):
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    return {
        "label": label, "model": model_name, "condition": condition,
        "threshold": threshold,
        "Recall": recall_score(y_test, pred, zero_division=0),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "F2": fbeta_score(y_test, pred, beta=2, zero_division=0),
        "AP": average_precision_score(y_test, prob),
        "AUROC": roc_auc_score(y_test, prob),
        "TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn),
    }

# 기준 6개 모델: 최초 설정 그대로 전체 Train에 학습, 공통 threshold=0.5.
for name in ["LR", "DT", "RF", "ET", "LGB", "XGB"]:
    print(f"기준 {name} Recall 평가", flush=True)
    baseline_pipe = Pipeline([
        ("clean", GMSCPreprocessor(use_log=True)),
        ("scale", StandardScaler()),
        ("model", clone(models[name])),
    ])
    baseline_pipe.fit(X_train, y_train)
    positive_col = list(baseline_pipe.classes_).index(1)
    prob = baseline_pipe.predict_proba(X_test)[:, positive_col]
    recall_rows.append(recall_metrics(
        name, name, prob, 0.5, "Baseline, threshold=0.5"
    ))

# 최종 모델은 이미 저장된 Test 확률을 재사용한다. Test에서 재튜닝하지 않는다.
frozen_predictions = pd.read_csv("final_test_predictions.csv")
assert np.array_equal(frozen_predictions["row_index"].to_numpy(), X_test.index.to_numpy())
assert np.array_equal(frozen_predictions["y_true"].to_numpy(), y_test.to_numpy())
with open("final_config.json", encoding="utf-8") as f:
    frozen = json.load(f)
frozen_prob = frozen_predictions["prob"].to_numpy()

recall_rows.append(recall_metrics(
    "Tuned XGB", "XGB", frozen_prob, 0.5, "Tuned, threshold=0.5"
))
recall_rows.append(recall_metrics(
    "Tuned XGB + F2 threshold", "XGB", frozen_prob,
    float(frozen["threshold"]), "Tuned, Train OOF F2 threshold"
))

recall_comparison = pd.DataFrame(recall_rows)
recall_comparison.to_csv(recall_out / "recall_comparison.csv", index=False)
display(recall_comparison)

# 모델 차이와 threshold 효과를 두 개의 그래프로 구분한다.
fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [2, 1]})
baseline_and_tuned = recall_comparison.iloc[:7]
axes[0].bar(
    baseline_and_tuned["label"], baseline_and_tuned["Recall"],
    color=["#193449"] * 6 + ["#199C9C"],
)
axes[0].set_title("Model comparison: common threshold = 0.5")
axes[0].tick_params(axis="x", rotation=40)
axes[0].set_ylabel("Test Recall")

final_two = recall_comparison.iloc[6:]
axes[1].bar(
    ["Tuned XGB\\nthreshold=0.5", "Tuned XGB\\nOOF F2 threshold"],
    final_two["Recall"], color=["#199C9C", "#DA9C44"],
)
axes[1].set_title("Threshold effect: same tuned model")
for ax in axes:
    ax.set_ylim(0, 1)
    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                f"{bar.get_height():.1%}", ha="center", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(recall_out / "recall_comparison.png", dpi=200, bbox_inches="tight")
plt.show()
'''
ast.parse(source)

if '--run' not in sys.argv:
    assert not any(heading in ''.join(c['source']) for c in nb['cells'])
    for kind, text in [
        ('markdown', heading + '\n\n- 6개 기준 모델과 튜닝 XGB는 threshold=0.5로 비교한다.\n- 튜닝 XGB의 Train OOF F2 threshold 결과는 별도 표시한다.\n- Recall만으로 비교하면 오탐 비용을 숨길 수 있어 Precision·F2·FP도 함께 저장한다.\n- 기존 Test를 확인한 뒤 추가한 설명용 비교이며 최종 모델 재선정에 사용하지 않는다.\n'),
        ('code', source),
    ]:
        c = dict(cell_type=kind, id=uuid4().hex[:8], metadata={}, source=text.splitlines(keepends=True))
        if kind == 'code':
            c.update(execution_count=None, outputs=[])
        nb['cells'].append(c)
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print('Recall comparison cells appended.')
else:
    import os
    os.chdir(ROOT)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.base import BaseEstimator, TransformerMixin, clone
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import recall_score, precision_score, fbeta_score, average_precision_score, roc_auc_score
    from imblearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    sys.path.insert(0, str(ROOT))
    from gmsc_preprocessing import GMSCPreprocessor
    SEED = 33
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        text = ''.join(cell['source'])
        tree = ast.parse(text)
        for item in tree.body:
            if isinstance(item, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'models' for t in item.targets):
                exec(compile(ast.Module(body=[item], type_ignores=[]), '<models>', 'exec'))
    data = pd.read_csv('givemesomecredit.csv').rename(columns={
        'SeriousDlqin2yrs': 'Dlq2yrs', 'RevolvingUtilizationOfUnsecuredLines': 'RevUtil',
        'age': 'Age', 'NumberOfTime30-59DaysPastDueNotWorse': 'PastDue_30-59',
        'MonthlyIncome': 'Income', 'NumberOfDependents': 'Dependents',
        'CombinedDefaulted': 'CombDefault', 'CombinedCreditLoans': 'CombCredit',
    })
    split = np.load('split_indices.npz')
    X_train = data.loc[split['train_idx']].drop(columns='Dlq2yrs')
    y_train = data.loc[split['train_idx'], 'Dlq2yrs']
    X_test = data.loc[split['test_idx']].drop(columns='Dlq2yrs')
    y_test = data.loc[split['test_idx'], 'Dlq2yrs']
    def display(value):
        print(value.to_string(index=False), flush=True)
    exec(compile(source, '<recall comparison>', 'exec'))
