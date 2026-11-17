"""실험 기록으로 발표 내용과 7분 대본을 구성한다. 표·차트의 숫자는 파일에서 가져온다."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'submission'
e = json.loads((OUT / 'presentation_evidence.json').read_text(encoding='utf-8'))
# pandas의 기본 CSV 파서가 문자열 None을 결측으로 읽은 기존 집계도 복구한다.
for row in e['balance']:
    if not isinstance(row['balance'], str):
        row['balance'] = 'None'
(OUT / 'presentation_evidence.json').write_text(json.dumps(e, ensure_ascii=False, indent=2), encoding='utf-8')
slides = []

def add(title, subtitle='', body='', chart=None, table=None, notes='', seconds=0):
    slides.append(dict(title=title, subtitle=subtitle, body=body, chart=chart, table=table,
                       notes=notes, seconds=seconds))

def chart(categories, series, max_value=0.4):
    return dict(categories=categories, series=series, maximum=max_value)

def series(name, values):
    return dict(name=name, values=values)

add('GMSC 연체 예측', '불균형 데이터에서 모델·PCA·보정 방법 비교',
    '주 지표 AP로 모델 선택\nTrain OOF로 F2 기준 임계값 결정\n격리한 Test로 최종 평가',
    notes='안녕하세요. GMSC 데이터를 이용한 연체 예측 과정을 발표하겠습니다. 이번 과제에서는 모델의 종류뿐 아니라 PCA와 불균형 보정이 실제로 도움이 되는지 비교했습니다. 최종적으로는 연체자를 더 찾기 위해 임계값을 조정하고, 별도로 보관한 Test 데이터에서 성능을 평가했습니다.', seconds=20)

d = e['data']
add('추진 배경: 양성 비율 6.68%', '정확도만으로 연체 탐지 성능을 판단하기 어려운 데이터',
    f"전체 {d['rows']:,}건 · 원본 특성 8개\n모두 0으로 예측해도 정확도 93.32%\n주 지표 AP · 보조 지표 AUROC\n최종 결정 성능은 F2·Recall·Precision으로 확인",
    table={'headers': ['Target', '건수', '비율'], 'rows': [
        ['0: 연체 미발생', f"{d['negative']:,}", '93.32%'],
        ['1: 90일 이상 연체 또는 악화', f"{d['positive']:,}", '6.68%']]},
    notes='전체 15만 건 중 양성은 약 만 건으로 6.68퍼센트입니다. 따라서 모든 사람을 연체가 없다고 예측해도 정확도가 93.32퍼센트입니다. 이 때문에 정확도 대신 AP를 주요 비교 기준으로 삼았고 AUROC를 함께 확인했습니다. 최종적으로 예측을 실제 0과 1로 바꾼 뒤에는 F2와 재현율, 정밀도를 보고 판단했습니다. Target은 90일 이상 연체 또는 그보다 심각한 상황을 나타내는 데이터의 레이블입니다.', seconds=40)

add('추진 전략: Test 격리와 단계별 비교', '모든 후보는 같은 Train과 5-fold 분할로 비교',
    '1   Stratified 80:20 분리 → Train 120,000 / Test 30,000\n2   Train EDA → 특수값 처리·로그 변환·스케일링\n3   6개 모델 → XGB·LGB·RF·LR 선정\n4   Original vs PCA → Original 유지\n5   보정 4조건 → 보정 없는 XGB 선정\n6   Train CV 튜닝 → OOF 임계값 → 최초 Test 평가',
    notes='먼저 전체 데이터를 클래스 비율을 유지하면서 8대 2로 나누었습니다. Train은 12만 건, Test는 3만 건이며 개발 과정에서는 Train만 사용했습니다. 전처리의 중앙값과 스케일링 값, PCA는 각 fold의 학습 부분에서만 계산했습니다. 연체 횟수의 96과 98은 정상값 중앙값으로 치환하고, 원래 특수값이었다는 정보를 별도 flag로 남겼습니다. 그래서 모델의 입력은 9개입니다. 같은 분할에서 모델, PCA, 보정 방법을 순서대로 비교한 다음, Train의 OOF 예측으로 임계값을 정하고 Test를 평가했습니다.', seconds=55)

b = sorted(e['baseline'], key=lambda r: r['AP_mean'], reverse=True)
add('모델 비교: XGB의 평균 AP가 가장 높음', '6개 모델에서 XGB·LGB·RF·LR을 후속 비교 대상으로 선정',
    '동일한 공통 전처리\n불균형 보정 없음\nAP 평균으로 순위 선정\n차이의 유의성은 미검정',
    chart=chart([r['model'] for r in b], [series('CV AP', [r['AP_mean'] for r in b])]),
    notes='기준 모델은 로지스틱 회귀, 의사결정나무, 랜덤 포레스트, 엑스트라 트리, LightGBM, XGBoost 여섯 가지입니다. AP 평균은 XGBoost가 약 0.350으로 가장 높았고, LightGBM이 약 0.341로 뒤를 이었습니다. 이후에는 두 부스팅 모델에 더해 비교용 선형 모델 LR과 배깅 모델 RF를 남겼습니다. 여기서 점수 차이를 통계적으로 유의한 우열이라고 주장하지는 않았습니다. 동일 조건에서 관측한 개발 성능 순위로 해석했습니다.', seconds=50)

names = ['XGB', 'LGB', 'RF', 'LR']
f = {(r['model'], r['feature']): r for r in e['feature']}
add('PCA: 9개에서 8개로 축소했지만 AP 하락', '입력 차원 감소보다 예측 정보 손실 가능성을 검토',
    '설명분산 95% 유지\n네 모델 모두 AP 하락\nPCA는 Target 미사용\nOriginal 유지',
    chart=chart(names, [series(k, [f[(n, k)]['AP_mean'] for n in names]) for k in ['Original', 'PCA']]),
    notes='다음은 원본 표현과 PCA의 비교입니다. 원본은 공통 전처리 후의 9개 특성이고, PCA는 설명분산 95퍼센트를 유지하도록 설정해 8개 성분으로 줄었습니다. 결과는 네 모델 모두 AP가 낮아졌습니다. PCA는 Target을 보지 않고 분산을 보존하기 때문에 작은 분산의 성분에도 예측에 필요한 정보가 있었을 가능성이 있습니다. 다만 이것을 확정 원인이라고 할 수는 없고, 이번 설정에서는 성능 개선에 도움이 되지 않았다는 결론으로 원본 표현을 유지했습니다.', seconds=45)

bal = {(r['model'], r['balance']): r for r in e['balance']}
add('불균형 보정: 모든 모델에 같은 효과는 없음', '보정 없는 XGB가 평균 AP 0.3498로 최상위',
    '학습 fold별 IR 계산\nLR·LGB: IR로 AP 개선\nXGB·RF: 보정 없음 우세\nNearMiss: 모두 AP 하락',
    chart=chart(names, [series(k, [bal[(n, k)]['AP_mean'] for n in names])
                        for k in ['None', 'IR', 'SMOTE', 'NearMiss']]),
    notes='불균형 보정은 없음, IR 가중치, SMOTE, NearMiss 네 조건입니다. IR은 각 fold의 학습 데이터에서 계산했습니다. SMOTE와 NearMiss는 학습 클래스 수를 1대 1로 맞췄고 검증 데이터는 건드리지 않았습니다. LR과 LightGBM은 IR 가중치로 AP가 개선됐지만, XGBoost와 RF는 보정 없는 조건이 더 높았습니다. NearMiss는 네 모델에서 모두 낮은 AP를 보였습니다. 따라서 불균형 데이터라고 반드시 리샘플링해야 하는 것은 아니며, 이번 비교에서는 보정 없는 XGBoost를 선택했습니다.', seconds=50)

default, final = e['frozen_test']
c = e['frozen_config']
add('최초 고정 모델의 Test 성능', f"XGB 250 trees · depth 3 · learning rate 0.05 · threshold {c['threshold']:.4f}",
    f"Test 30,000건\nAP {final['AP']:.4f} · AUROC {final['AUROC']:.4f}\n\nRecall {default['Recall']:.1%} → {final['Recall']:.1%}\nPrecision {default['Precision']:.1%} → {final['Precision']:.1%}\nF2 {default['F2']:.3f} → {final['F2']:.3f}\n\n임계값: Train OOF에서 선택",
    table={'headers': ['실제 / 예측', '0', '1'], 'rows': [
        ['0', f"{final['TN']:,}", f"{final['FP']:,}"],
        ['1', f"{final['FN']:,}", f"{final['TP']:,}"]]},
    notes='최초 튜닝은 트리 수, 깊이, 학습률의 12개 조합을 비교했습니다. 선택된 모델은 트리 250개, 깊이 3, 학습률 0.05입니다. 이 설정의 Train OOF 예측에서 F2를 최대화하는 임계값은 약 0.1004였습니다. 이를 고정해 Test에 적용한 결과 AP는 0.3488, AUROC는 0.8580입니다. 기본 임계값 0.5에서는 Recall이 약 10.2퍼센트였지만, 선택 임계값에서는 66.6퍼센트로 높아졌습니다. F2도 0.122에서 0.505로 높아졌습니다. 대신 Precision은 57퍼센트에서 25.7퍼센트로 낮아졌습니다. 혼동행렬에서 실제 양성 2,005명 중 1,335명을 찾았고 670명을 놓쳤습니다. 오탐은 3,859명입니다. AP와 AUROC는 확률 순위 지표라 임계값을 바꿔도 변하지 않습니다. 이 Test 성적은 뒤에서 설명할 추가 탐색 모델과 구분되는 최초 고정 모델의 기록입니다.', seconds=100)

add('의사결정 지원과 한계', '연체 탐지 확대와 오탐 증가를 함께 제시',
    '발견한 양성: 1,335 / 2,005명\n양성으로 예측한 5,194명 중 실제 양성은 25.7%\n\n이번 조건에서는 Original + 보정 없는 XGB 선정\n단계적 탐색이므로 전체 조합의 최적성은 보장하지 않음\n합성 표본의 이산값 보간·범주 가공 규칙에 한계\n추가 검증: 새 독립 데이터·비용 기준·특성 정의 확인',
    notes='이 결과는 연체자를 더 찾는 대신 추가 확인해야 할 대상도 늘어난다는 점을 보여줍니다. 양성으로 예측한 5,194명 중 실제 양성은 1,335명으로 약 25.7퍼센트입니다. 따라서 성능을 설명할 때 Recall만 강조하지 않고 오탐 규모도 함께 제시해야 합니다. 이번 과제에서는 원본 표현과 보정 없는 XGBoost가 적절한 후보였지만, 모든 피처와 보정 조합을 탐색한 것은 아닙니다. 범주형으로 가공된 변수의 정확한 정의와 일반 SMOTE의 이산값 보간도 한계입니다. 또한 Test를 확인한 뒤 마지막 추가 튜닝을 수행했기 때문에, 그 결과는 Train CV 탐색으로 따로 기록하고 기존 Test 점수를 교체하지 않았습니다. 후속 모델은 새로운 독립 데이터에서 검증해야 합니다. 감사합니다.', seconds=60)

fu = e['followup']
add('부록: 마지막 추가 튜닝은 Train CV만 평가', 'Test를 확인한 후의 탐색이므로 최초 Test 성적과 구분',
    f"20개 후보 × 5-fold\n기존 최적 설정도 포함\n후속 CV AP {fu['cv_AP']:.4f}\n기존 대비 {fu['AP_delta']:+.4f}\n\n기존 최적 설정 유지\nTest는 재평가하지 않음\n추가 성능 향상 없음",
    table={'headers': ['파라미터', '후속 선택값'], 'rows': [
        [k.replace('model__', ''), str(v)] for k, v in fu['best_params'].items()]},
    notes='발표 후 질문이 있으면 사용할 추가 탐색 기록입니다. 기존 최적 설정을 포함한 20개 후보를 같은 Train CV로 비교했고 깊이, 트리 수, 학습률 외에 min_child_weight, subsample, reg_lambda를 탐색했습니다. 후속 결과는 Train CV 개발 성능이며, 기존 Test를 반복해서 선택 기준으로 사용하지 않았습니다. CV 점수 상승을 독립적인 일반화 성능 향상으로 해석하지 않습니다.')

add('부록: 지표·재현성·근거', '',
    'AP: PR 곡선 기반 평균 정밀도, 사다리꼴 PR-AUC와 구분\nAUROC: 양성·음성의 순위 판별 성능\nF2: Recall을 더 중시하는 Precision·Recall 결합 지표\nBrier: 확률 예측 오차, 순수한 보정 오차만의 지표는 아님\n\nseed=33 · Stratified 80:20 · Train 5-fold\n원본 데이터와 가공 변수 정의의 확인이 필요\nXAI·유의성 검정은 이번 결과에 포함하지 않음',
    notes='수치 근거: baseline_summary.csv, feature_summary.csv, balance_summary.csv, final_test_metrics.csv, final_config.json, submission/followup_tuning_results.csv. 지표 설명: https://scikit-learn.org/stable/modules/model_evaluation.html . threshold: https://scikit-learn.org/stable/modules/classification_threshold.html . XGB: https://xgboost.readthedocs.io/en/stable/parameter.html . 논문 제목과 정확한 범주 가공 규칙은 제공되지 않아 임의로 기재하지 않았습니다.')

assert sum(s['seconds'] for s in slides[:8]) == 420
(OUT / 'deck_spec.json').write_text(json.dumps(slides, ensure_ascii=False, indent=2), encoding='utf-8')
script = ['GMSC 발표 대본 — 본문 8장, 목표 7분 / 부록 2장',
          '시간은 배분 목표입니다. 실제 낭독 속도로 1회 리허설 후 문장을 조절하세요.', '']
for i, s in enumerate(slides):
    script.extend([f"{i+1}. {s['title']} ({s['seconds']}초)" if s['seconds'] else f"{i+1}. {s['title']} (부록)", s['notes'], ''])
(OUT / '발표대본_7분.txt').write_text('\n'.join(script), encoding='utf-8')
summary = f'''GMSC 과제 결과 정리

1. 데이터: 150,000건, 원본 특성 8개. 특수값 flag 추가 후 입력 9개.
2. 분할: Stratified Train 120,000 / Test 30,000, seed=33. 개발은 Train 5-fold.
3. 모델: LR/DT/RF/ET/LGB/XGB 비교 후 XGB/LGB/RF/LR 후속 비교.
4. PCA: 설명분산 95% 기준 9→8차원. 네 모델 모두 AP 하락. 확정 원인은 미검증.
5. 보정: None/IR/SMOTE/NearMiss. 보정 없는 XGB가 AP 1위.
6. 최초 튜닝: 12조합. 250 trees / max_depth=3 / learning_rate=0.05.
7. Train OOF F2 기준 threshold={c['threshold']:.6f} 고정 후 최초 Test 평가.
8. 최초 Test: AP={final['AP']:.6f}, AUROC={final['AUROC']:.6f}, F2={final['F2']:.6f}, MCC={final['MCC']:.6f}, Brier={final['Brier']:.6f}.
   Precision={final['Precision']:.6f}, Recall={final['Recall']:.6f}. TN={final['TN']}, FP={final['FP']}, FN={final['FN']}, TP={final['TP']}.
9. 마지막 추가 튜닝: 20후보, 기존 최적 포함. Train CV AP={fu['cv_AP']:.6f} (기존 대비 {fu['AP_delta']:+.6f}).
   선택 설정={json.dumps(fu['best_params'], ensure_ascii=False)}.
   추가 탐색에서 Test는 재평가하지 않음. 기존 최적 설정이 다시 선택되어 기존 고정 모델의 Test 결과를 유지.
10. 한계: 단계적 선택 편향, CV fold 비독립성, 범주 가공 규칙 미확정, 일반 SMOTE 보간, 단일 데이터셋.
    검정·XAI는 수행한 것으로 쓰지 않는다. CV 차이를 통계적 유의성으로 주장하지 않는다.

제출 확인
- ass.ipynb의 기존 실행 결과 및 데이터 경로 확인
- GMSC_7min.pptx: 본문 8장+부록 2장, 발표자 노트 포함
- 발표대본_7분.txt로 실제 리허설
- 익명 발표자료이므로 필요한 이름·학번을 표지에 직접 입력
- 이후 Test 점수를 보고 설정을 다시 선택하지 않기
'''
(OUT / '결과정리_제출안내.txt').write_text(summary, encoding='utf-8')
print('Deck specification, script and summary prepared.')
