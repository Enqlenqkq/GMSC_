AUPRC -> Imbalanced Classification에 효과적
AUROC - X축: FPR / Y축: TPR (Recall)
AUPRC - X축: Recall / Y축: Precision

① PastDue_30-59의 특수 코드(96, 98) 분리

• 96과 98을 그대로 두면 트리 모델이나 선형 모델 모두 "연체를 98번 한 초고위험군"으로 잘못 학습합니다.
• 96과 98 여부를 나타내는 별도의 이진 플래그 변수(PastDue_special_flag)를 생성한 뒤, 원래 컬럼의 96, 98 값은 해당 컬럼의 중앙값(또는 최빈값인 0)이나 정상 최대 연체 횟수로 치환하세요.

② 극단적 비율 변수(RevUtil, DebtRatio) 상한선 제한(Clipping) 및 로그 변환

• DebtRatio가 수천~수십만에 달하는 이유는 과제 출제자가 결측치였던 Income을 중앙값(6600)으로 채우기 전에 비율이 계산되었기 때문입니다(소득이 0이거나 누락된 상태에서 부채액 자체가 비율로 기록된 GMSC 특유의 현상).
• 상한선 제한(Winsorizing / Clipping): 상위 99% 또는 99.5% 분위수(Quantile)를 기준으로 극단값을 잘라내어 모델의 가중치 왜곡을 막아야 합니다.
• 로그 변환(np.log1p): RevUtil, DebtRatio, Income 모두 왜도가 95~127로 매우 높습니다. 특히 로지스틱 회귀나 신경망을 함께 테스트한다면 np.log1p() 변환과 스케일링(RobustScaler)이 필수입니다.

③ Income의 대체 흔적 및 0값 처리

• Income의 50%(중앙값)가 정확히 6600.000인 점으로 미루어 보아, 기존 결측치 약 20%가 중앙값 6600으로 일괄 대체되었을 가능성이 높습니다.
• Income == 6600인 데이터와 Income == 0인 데이터를 구분하는 **플래그 변수(Income_is_zero, Income_is_median)**를 만들어 주면, 모델이 실제 소득 6,600달러인 사람과 결측 대체된 사람을 구분해 부도 확률을 더 정교하게 잡아냅니다.

④ 도메인 기반 상호작용 파생변수 추가

• 실질 월 부채 상환액 추정: DebtRatio가 정상 범위(예: 10 이하)일 때 DebtRatio \* Income을 계산해 실제 매달 나가는 고정 지출 규모를 파생변수로 만듭니다.
• 총 연체 강도: 정제된 PastDue_30-59와 기존 심각한 연체 여부인 CombDefault를 결합해 연체 이력 점수를 생성합니다.

3. 실전 적용 전처리 코드 스니펫

python
import numpy as np
import pandas as pd

def prepare_assignment_gmsc(df):
data = df.copy()

    # 1. PastDue_30-59 특수 코드(96, 98) 분리 및 치환
    data['PastDue_special_flag'] = data['PastDue_30-59'].isin().astype(int)
    # 정상 데이터의 최댓값 또는 중앙값(0)으로 치환
    normal_max = data.loc[~data['PastDue_30-59'].isin(), 'PastDue_30-59'].max()
    data['PastDue_30-59_clean'] = data['PastDue_30-59'].replace(, normal_max)

    # 2. Income 특수값(0 및 결측 대체 추정값 6600) 플래그 생성
    data['Income_is_zero'] = (data['Income'] == 0).astype(int)
    data['Income_is_6600'] = (data['Income'] == 6600).astype(int)

    # 3. 극단적 이상치 상한선 제한 (99.5% 분위수 기준 Clipping)
    for col in ['RevUtil', 'DebtRatio', 'Income']:
        upper_limit = data[col].quantile(0.995)
        data[f'{col}_clipped'] = np.clip(data[col], a_min=0, a_max=upper_limit)
        # 왜도 완화를 위한 로그 변환
        data[f'{col}_log'] = np.log1p(data[f'{col}_clipped'])

    # 4. 금융 도메인 파생변수
    data['Est_Monthly_Debt'] = data['DebtRatio_clipped'] * data['Income_clipped']
    data['Income_per_Person'] = data['Income_clipped'] / (data['Dependents'] + 1)
    data['Total_Risk_Score'] = data['PastDue_30-59_clean'] + (data['CombDefault'] * 3)

    return data
