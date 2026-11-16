AUPRC -> Imbalanced Classification에 효과적
AUROC - X축: False Positive Rate (\(\frac{FP}{FP+TN}\)) / Y축: True Positive Rate (Recall)
AUROC의 X축 FPR은 분모에 TN이 포함되어있어 GMSC같은 극소수 양성 클래스에선 양성 예측 실패에 대해 변동이 매우 미미함.
Class에 대해 Neg: 139974 / Pos: 10026인데, FPR이 0 ~ 0.0668
AUPRC - X축: Recall / Y축: Precision

Precision = TP / (TP + FP) => 양성 예측 중 실제 양성 비율
TPR(Recall) = TP / (TP + FN) => 실제 양성 중 찾아낸 비율
FPR = FP / (FP + TN) => 실제 음성 중 양성으로 잘못 판단 (1종 오류)
Prevalence = (TP + FN) / N => 전체 데이터에서 양성 비율 (IR)

AUROC의 무작위 예측 기준선은 항상 0.5지만, AUPRC의 기준선은 소수 클래스의 비율입니다. 현재 데이터셋의 경우 약 0.067(6.68%)이 기준선

④ 도메인 기반 상호작용 파생변수 추가

• 실질 월 부채 상환액 추정: DebtRatio가 정상 범위(예: 10 이하)일 때 DebtRatio \* Income을 계산해 실제 매달 나가는 고정 지출 규모를 파생변수로 만듭니다.
• 총 연체 강도: 정제된 PastDue_30-59와 기존 심각한 연체 여부인 CombDefault를 결합해 연체 이력 점수를 생성합니다.

NSA(Non-parametric Stacked Autoencoder)/PCA
Resampler - cWGAN / SMOTE / NearMiss
Model - LR/DT/ET/RF/GB/XGB
