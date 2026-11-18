"""학습 파이프라인과 Train OOF 임계값을 joblib로 저장한다."""
import copy
import json
from pathlib import Path

import joblib
import numpy as np
from gmsc_preprocessing import GMSCPreprocessor


def export_final_model(pipeline, config, X_sample, output='submission/final_model.joblib'):
    """기존 학습 상태를 보존하며 노트북 __main__ 클래스 의존성을 제거한다."""
    for param, value in config['best_params'].items():
        if pipeline.get_params()[param] != value:
            raise ValueError(f'고정 설정과 모델이 다릅니다: {param}')
    if pipeline.get_params()['model__scale_pos_weight'] != 1.0:
        raise ValueError('최종 모델은 보정 없는 XGB입니다.')

    portable = copy.deepcopy(pipeline)
    # 이미 실행한 커널에 남은 노트북 클래스도 같은 학습 상태의 모듈 클래스로 교체한다.
    clean = GMSCPreprocessor(use_log=pipeline.named_steps['clean'].use_log)
    clean.__dict__.update(copy.deepcopy(pipeline.named_steps['clean'].__dict__))
    portable.set_params(clean=clean)
    np.testing.assert_allclose(
        portable.predict_proba(X_sample), pipeline.predict_proba(X_sample),
        rtol=0, atol=0,
    )
    artifact = {
        'pipeline': portable,
        'threshold': float(config['threshold']),
        'input_columns': list(X_sample.columns),
        'positive_label': 1,
        'config': copy.deepcopy(config),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output, compress=3)
    loaded = joblib.load(output)
    np.testing.assert_allclose(
        loaded['pipeline'].predict_proba(X_sample),
        pipeline.predict_proba(X_sample), rtol=0, atol=0,
    )
    print(f'저장·재로드 검증 완료: {output.resolve()}')
    return loaded


def main():
    """고정된 최종 설정으로 Train만 재학습해 독립 실행 가능한 산출물을 만든다."""
    import os
    import pandas as pd
    from imblearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    root = Path(__file__).resolve().parent
    os.chdir(root)
    config = json.loads(Path('final_config.json').read_text(encoding='utf-8'))
    data = pd.read_csv('givemesomecredit.csv').rename(columns={
        'SeriousDlqin2yrs': 'Dlq2yrs', 'RevolvingUtilizationOfUnsecuredLines': 'RevUtil',
        'age': 'Age', 'NumberOfTime30-59DaysPastDueNotWorse': 'PastDue_30-59',
        'MonthlyIncome': 'Income', 'NumberOfDependents': 'Dependents',
        'CombinedDefaulted': 'CombDefault', 'CombinedCreditLoans': 'CombCredit',
    })
    split = np.load('split_indices.npz')
    X_train = data.loc[split['train_idx']].drop(columns='Dlq2yrs')
    y_train = data.loc[split['train_idx'], 'Dlq2yrs']
    pipeline = Pipeline([
        ('clean', GMSCPreprocessor(use_log=config['use_log'])),
        ('scale', StandardScaler()),
        ('model', XGBClassifier(tree_method='hist', eval_metric='logloss',
                                scale_pos_weight=1.0, random_state=config['seed'], n_jobs=1)),
    ])
    pipeline.set_params(**config['best_params'])
    pipeline.fit(X_train, y_train)
    loaded = export_final_model(pipeline, config, X_train.iloc[:32])

    # Test는 학습이나 선택에 쓰지 않고 기존 예측을 그대로 재현하는지만 검증한다.
    X_test = data.loc[split['test_idx']].drop(columns='Dlq2yrs')
    expected = pd.read_csv('final_test_predictions.csv')
    assert np.array_equal(expected['row_index'].to_numpy(), X_test.index.to_numpy())
    pos = list(loaded['pipeline'].classes_).index(loaded['positive_label'])
    prob = loaded['pipeline'].predict_proba(X_test)[:, pos]
    np.testing.assert_allclose(prob, expected['prob'], rtol=0, atol=1e-6)
    assert np.array_equal((prob >= loaded['threshold']).astype(int), expected['prediction'])
    print('기존 최종 Test 예측과 일치합니다. 설정·임계값 변경 없음.')


if __name__ == '__main__':
    main()
