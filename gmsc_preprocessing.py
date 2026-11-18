import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# 전처리기: 처리 규칙을 한 객체로 묶어 Pipeline과 CV에서 재사용한다.
# BaseEstimator: 설정 조회·변경 및 CV에서 객체 복제를 지원한다.
# TransformerMixin: fit 후 transform을 호출하는 fit_transform 등을 제공한다.
class GMSCPreprocessor(BaseEstimator, TransformerMixin):
    # 생성할 때는 설정만 저장한다. 중앙값 계산은 fit에서 수행한다.
    def __init__(self, use_log=True):
        # True면 로그 변환 적용, False면 특수값 처리만 수행한다.
        self.use_log = use_log

    # fit: 현재 fold의 학습 데이터에서 변환에 필요한 값을 구해 저장한다.
    # y는 Pipeline 호환용 인자이며 이 전처리에서는 정답을 사용하지 않는다.
    def fit(self, X, y=None):
        col = "PastDue_30-59"
        # isin은 96 또는 98인 행을 True로 표시하고, ~는 True/False를 뒤집는다.
        # 따라서 normal은 특수값이 아닌 행을 선택하는 마스크다.
        normal = ~X[col].isin([96, 98])
        # loc[행 조건, 컬럼]으로 정상값만 선택해 중앙값을 구한다.
        # 이름 끝의 _는 fit에서 학습한 속성이라는 sklearn 관례다.
        self.pastdue_median_ = X.loc[normal, col].median()
        # 데이터를 반환하는 것이 아니라 학습을 마친 객체 자신을 반환한다.
        return self

    # transform: 새 통계를 계산하지 않고 fit에서 저장한 규칙으로 데이터를 변환한다.
    # 학습·검증 데이터에 같은 규칙을 적용하므로 검증 데이터의 정보가 새지 않는다.
    def transform(self, X):
        # 원본 DataFrame을 덮어쓰지 않도록 복사본에서 작업한다.
        result = X.copy()
        col = "PastDue_30-59"
        special = result[col].isin([96, 98])

        # 치환 전에 특수값 여부를 별도 특성으로 보존한다(True=1, False=0).
        result["PastDue_special_flag"] = special.astype(int)
        # 특수값이 있는 행만 학습 데이터에서 구한 중앙값으로 바꾼다.
        result.loc[special, col] = self.pastdue_median_

        if self.use_log:
            cols = ["RevUtil", "DebtRatio", "Income"]
            # log1p(x)=log(1+x): 0도 처리할 수 있고 큰 값 사이의 간격을 줄인다.
            # 현재 EDA에서 이 세 컬럼의 최솟값은 0이다. 왜도가 완전히 사라지는 것은 아니다.
            result[cols] = np.log1p(result[cols])

        # 컬럼명과 행 인덱스를 유지한 변환 결과를 반환한다.
        return result
