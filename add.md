# 참고 아이디어와 실제 수행 범위

최종 구현은 `ass.ipynb`, 보완 근거는 `submission/supplementary/`를 기준으로 확인한다.
아래 도메인 아이디어는 제안 단계이며 모두 최종 모델에 적용한 것은 아니다.

AP와 사다리꼴 적분 PR-AUC는 계산 방식이 다르다. 과제의 주 지표는 AP다.
AUROC의 X축은 FPR, Y축은 TPR(Recall)이며 FPR 범위는 0~1이다.
양성 비율 약0.0668은 FPR 상한이 아니라 무작위 순위 PR 평가의 기준 비율이다.

Precision = TP / (TP + FP) => 양성 예측 중 실제 양성 비율
TPR(Recall) = TP / (TP + FN) => 실제 양성 중 찾아낸 비율
FPR = FP / (FP + TN) => 실제 음성 중 양성으로 잘못 판단 (1종 오류)
Prevalence = (TP + FN) / N => 전체 데이터에서 양성 비율
IR = 음성 수 / 양성 수 => 약13.961

AUROC의 무작위 예측 기준선은 항상 0.5지만, AUPRC의 기준선은 소수 클래스의 비율입니다. 현재 데이터셋의 경우 약 0.067(6.68%)이 기준선

## 도메인 기반 파생변수 후보 — 미수행

- `DebtRatio * Income`: 두 변수의 원래 단위·분모와 결측 대체 이력이 확인돼야 한다.
  Income=0 또는 대체된 소득을 곱해 부채가 0이라고 잘못 해석할 수 있으므로 현재 실험에서 제외했다.
- `PastDue_30-59 + CombDefault`: CombDefault의 범주 가공 규칙을 모르는 상태에서
  실제 횟수처럼 합산하지 않는다. 현재 실험에서 제외했다.
- 두 후보는 성능 개선이 없어서 제거한 것이 아니라, 의미 검증을 하지 않아 구현하지 않은 것이다.

## 소득·부채비율의 보완 진단

- 관측 `Income==0` 그룹과 DebtRatio 구간의 Train 연체율·표본 수를 확인했다.
- `Zero_Income_flag`는 사후 Train CV 전처리 ablation에만 추가했다. 최종 모델에는 넣지 않았다.
- 큰 DebtRatio가 무소득·과거 결측과 관련될 가능성은 있으나, 현재 가공 CSV만으로
  모든 극단값이 달러 절대액이라고 확정하거나 원래 결측 행을 복원할 수는 없다.
- `Income==6600`은 관측값 그룹이다. 원래 결측이었다는 표시로 취급하지 않는다.
- 결과는 `submission/supplementary/train_group_risk.csv`와 `preprocessing_ablation.csv`에 저장했다.

## 실제 적용한 범위

- 피처 표현: 공통 전처리 Original9 / PCA8. 도메인 파생변수 결합·RFE·chi2 선택은 미수행.
- 보정: 없음 / IR / SMOTE / NearMiss. NSA·cWGAN은 제외했다.
- 기준 모델: LR / DT / ET / RF / LGB / XGB.
- 중요도는 최종 모델 SHAP으로 설명하며 특성 선택의 인과 근거로 쓰지 않는다.
- 기준 XGB/LGB/IR 간 AP 차이는 사후 고정 모델 Test bootstrap으로 확인했다.
  CV 평균의 신뢰구간이나 재학습·선택 불확실성을 포함한 구간은 아니다.
