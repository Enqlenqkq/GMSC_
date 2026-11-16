## Lab 1

### Descriptive Statistics

- Observation of Raw Data
    - `head()`, `tail()`
- Dimensions of Data
    - `shape`
- Data Type for Each Attribute
    - `dtypes`
- Statistical Summary
    - `describe()`
- Class Distribution
    - `data.groupby('Dlq2yrs').size()
- Correlations between Attributes
    - `corr()`
- Skew of Univariate Distributions
    - `skew()`

## Lab 2

### Visualisation

- Univariate
    - Histogram
        - `hist()`
    - Density Plot
        - `plot(kind='density', subplots=True, layout=(3, 3), sharex=False)`
    - Box Plot
        - `plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False)`
- Multivariate
    - Correlation Matrix Plot

        ```python
        corr = data.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
        plt.show();
        ```

    - Scatter Plot Matrix

        ```python
        from pandas.plotting import scatter_matrix

        scatter_matrix(data, figsize=(14, 14), diagonal='hist')
        plt.tight_layout()
        plt.show();
        ```

### Data Preparation

- Min Max Scaler
    - `MinMaxScaler()` (Rescale: 0 ~ 1)
- Standardisation
    - `StandardScaler()` (Rescale: mean=0, std=1)
- Normalization
    - `Normalizer()` (Rescale: Unit Vector)
- Binarization
    - `Binarizer()` (Rescale: 0 <== x < Threshold < x ==> 1)

## Lab 3

### Feature Selection

- Select K Score Best - chi2

    ```python
    from sklearn.feature_selection import SelectKBest, chi2

    chi2test = SelectKBest(score_func=chi2, k=4)
    fit = chi2test.fit(X,Y)

    score = pd.DataFrame({'Feature': data.columns[1:], 'Chi2_Score': fit.scores_})
    score.set_index('Feature')

    features = fit.transform(X)
    print(features[:5])
    ```

- Recursive Feature Elimination, RFE

    ```python
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(solver='liblinear')
    rfe = RFE(model, 3)
    fit = rfe.fit(X, Y)
    print(f"Num Features: {fit.n_features_}")
    print(f"Selected Features: {fit.support_}")
    print(f"Feature Ranking: {fit.ranking_}")
    ```

- Data Reduction: Principal Component Analysis, PCA

    ```python
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    fit = pca.fit(X)

    print(f"Explained Variance: {fit.   explained_variance_ratio_}")
    print(fit.components_)
    ```

- Feature Importance - ExtraTreesClassifier

    ```python
    from sklearn.ensemble import    ExtraTreesClassifier

    model = ExtraTreesClassifier()
    model.fit(X, Y)
    print(model.feature_importances_)
    ```

### Resampling

- Split into Train and Test Sets

    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    test_size = 0.33 # Train : Test = 0.67 : 0.33
    seed = 7
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,   test_size=test_size, random_state=seed)
    model = LogisticRegression(solver='liblinear')
    model.fit(X_train, Y_train)
    result = model.score(X_test, Y_test)
    print(f"Accuracy: {result * 100.0}");
    ```

- K-fold Cross-Validation

    ```python
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.linear_model import LogisticRegression

    num_folds = 10
    seed = 7
    kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
    model = LogisticRegression(solver='liblinear')
    results = cross_val_score(model, X, Y, cv=kfold)
    print(f"Accuracy: {results.mean() * 100.0} ({results.std() * 100.0})")
    ```

- Leave-One-Out Cross-Validation

    ```python
    from sklearn.model_selection import LeaveOneOut, cross_val_score
    from sklearn.linear_model import LogisticRegression

    loocv = LeaveOneOut()
    model = LogisticRegression(solver = 'liblinear')
    results = cross_val_score(model, X, Y, cv=loocv)
    print(f"Accuracy: {results.mean()*100.0} ({results.std()*100.0})")
    ```

- Repeated Random Test-Train Splits

    ```python
    from sklearn.model_selection import ShuffleSplit, cross_val_score
    from sklearn.linear_model import LogisticRegression

    n_splits = 10
    test_size = 0.33
    seed = 7
    kfold = ShuffleSplit(n_splits=n_splits, test_size=test_size,  random_state=seed)
    model = LogisticRegression(solver = 'liblinear')
    results = cross_val_score(model, X, Y, cv=kfold)
    print(f"Accuracy: {results.mean()*100.0} ({results.std ()*100.0})")
    ```

## Lab 4

### Evaluation Metrics

- Classification Metrics
    - Classification Accuracy

        ```python
        from sklearn.model_selection import KFold, cross_val_score
        from sklearn.linear_model import LogisticRegression

        kfold = KFold(n_splits=10, random_state=7, shuffle=True)
        model = LogisticRegression(solver='liblinear')
        scoring = 'accuracy'    # Accuracy
        results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
        print(f"Accuracy: {results.mean() * 100.0} ({results.std() * 100.0})")
        ```

    - Logarithmic Loss: '손실'이므로 값이 작을수록 좋음

        ```python
        from sklearn.model_selection import KFold, cross_val_score
        from sklearn.linear_model import LogisticRegression

        kfold = KFold(n_splits=10, random_state=7, shuffle=True)
        model = LogisticRegression(solver='liblinear')
        scoring = 'neg_log_loss'    # LogLoss
        results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
        print(f"Accuracy: {results.mean() * 100.0} ({results.std() * 100.0})")
        ```

    - AUROC: ROC (Y: Positive 예측 성공률, X: Positive 예측 실패율) 곡선 아래 면적. 0 ~ 1이고, 1에 가까울수록 좋음

        ```python
        from sklearn.model_selection import KFold, cross_val_score
        from sklearn.linear_model import LogisticRegression

        kfold = KFold(n_splits=10, random_state=7, shuffle=True)
        model = LogisticRegression(solver='liblinear')
        scoring = 'roc_auc'    # AUROC
        results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
        print(f"Accuracy: {results.mean() * 100.0} ({results.std() * 100.0})")
        ```

    - Confusion Matrix: X축은 예측, Y축은 실제.

        ```python
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import confusion_matrix

        test_size = 0.33
        seed = 7
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

        model = LogisticRegression(solver='liblinear')
        model.fit(X_train, Y_train)
        predicted = model.predict(X_test)
        matrix = confusion_matrix(Y_test, predicted)
        print(matrix);
        ```

    - Classification Report: 정밀도(Precision), 재현율(Recall), F1-점수(F1-score), 지지도(Support) 표시

        ```python
        from sklearn.metrics import classification_report

        report = classification_report(Y_test, predicted)
        report
        ```

### Shortlisting Algorithms

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machines (SVM)
- Classification and Regression Trees(CART or Decision Tree)

### Comparing ML Algorithms

- Logistic Regression (LR)
- Linear Discriminant Analysis (LDA)
- K-Nearest Neighbors (KNN)
- Classification and Regression Tree (CART)
- Naive Bayes (NB)
- Support Vector Machines (SVM)

## Lab 5

### Pipelining

- PreProcessing + Feature Selection + Model -> Evaluation

### Ensemble

- Bagging Algorithms (Bootstrap Aggregation): 훈련 데이터셋의 서로 다른 하위 샘플로부터 여러 모델을 구축. 분산이 높은 알고리즘에 효과적.
    - Bagged Decision Trees
    - Random Forest: 개별 분류기 사이의 상관관계를 줄임. 매 분할마다 무작위 특성의 하위 집합만을 고려
- Boosting Algorithms: 여러 모델 구축, 이전 모델의 예측 오류를 다음 모델이 수정하는 방식
    - AdaBoost: 쉽게 맞추는 인스턴스는 덜 집중하고, 어려운 인스턴스에 집중하게 함
- Voting Ensemble: 여러 모델 구축, 예측값들을 평균 등의 단순 통계를 이용해 결합

### Tuning

- Grid Search Parameter Tuning: 그리드 형태로 명시된 모든 파라미터 조합에 대해 모델을 구축하고 평가
- Random Search Parameter Tuning

### Saving and Loading Models

- Pickle
- Joblib

## Lab 6

### Keras and Tensorflow

### Deep Learning with MLP

- Load Data
- Define Model
    - Dense()
- Compile Model
- Evaluate Model
    - Evaluate Model
    - Save Weights
- Data Splitting
    - Automatic Verification Dataset
    - Manual Verification Dataset
- Manual K-Fold Cross-Validation

### Use Keras with Scikit-Learn

- Evaluate Models with Cross-Validation
- Grid Search Deep Learning Model Parameters

## GMSC Model Flow 및 노트북 가이드

1. 데이터 로드
2. 데이터 분할 - stratify=y로 Train/Test 분리 및 Test 격리, 분할 인덱스 저장
3. Train 데이터 시각화 및 실험 기준 설정 - 주 지표 AP, threshold 목적 사전 설정

----------- 각 cv fold 내에서만 fit, 동일한 cv 분할 -----------

4. 결측치·왜도·특수값 확인/처리 - 유효한 위험 신호인지 확인 후 처리 후보 비교
   PastDue_30-59(96,98) -> 의미 확인, 정상값의 학습 중앙값으로 치환, flag 남기기
   RevUtil, DebtRatio, Income -> np.log1p() 변환 / 변환 안함 비교
5. 데이터 스케일링
6. 특성 선택·추출·확장 - 원본 / PCA 추출 / 원본 + PCA 비교
7. 불균형 보정 - 가중치와 리샘플링을 별도 조건으로 비교
   리샘플링 없음 / 가중치(IR) / SMOTE / NearMiss 비교

8. 모델 학습·튜닝·선택 - LR/DT/ET/RF/GB/XGB, CV 평균 AP 기준

----------- 모델 학습·튜닝까지 CV 내부에서 수행 -----------

9. 평가 지표 - AP(주 지표), AUROC / F1, F2, MCC, Precision, Recall / Brier(확률 예측 오차)
10. Threshold 결정 - 선택한 설정의 Train OOF 예측에서 F2 최대화, Test 사용 금지
11. 최종 재학습·Test 평가·성능 차이 분석 - 사전 지정 모델 간 paired bootstrap 신뢰구간
12. XAI 분석 - 최종 모델에 맞는 SHAP explainer 사용

### 데이터 사용 범위

위 Lab 1~6은 수업 참고 자료다. GMSC 실행에는 이 가이드를 적용한다.

| 데이터 | 용도 | 적용 원칙 |
|---|---|---|
| 전체 원본 | 스키마·ID·중복 등 기본 검증 및 최초 분할 | 동일 고객·중복이 양쪽에 걸치면 그룹 분할 검토 |
| Train 전체 | EDA, 개발, 최종 재학습 | EDA로 비교 후보를 정하고 효과는 CV로 확인 |
| Fold 학습 부분 | 전처리·추출기·모델 학습 및 리샘플링 | 모든 통계와 학습 파라미터는 여기서 계산 |
| Fold 검증 부분 | 학습된 변환 적용 및 성능 평가 | 리샘플링하지 않으며 조기 종료용으로 재사용하지 않음 |
| Test | 설정 확정 후 최종 평가 | 특성·모델·threshold 결정에 사용하지 않음 |

- 기본안: Train/Test=80/20, seed=42. 실제 비율과 seed는 실험 시작 전에 확정한다.
- Holdout 분리는 `train_test_split(..., stratify=y)`, Train 내부 CV는
  `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`로 구분한다.
- 동일 비교 조건에는 동일 fold 인덱스를 사용한다. 매 fold마다 모든 학습 객체를 새로 만든다.

### Fold 내부 처리 순서

```text
Fold 학습 부분
  → 결측치·특수값 처리 → 선택적 log1p → 스케일링
  → 특성 선택·추출·결합 → 리샘플링 또는 가중치 설정 → 모델 학습

Fold 검증 부분
  → 학습된 전처리·특성 변환만 적용 → 예측·평가
```

- 중앙값·clipping 경계·스케일러·PCA는 fold 학습 부분에서만 fit한다.
- 로그 변환 전 값의 범위를 확인한다. 트리 모델의 변환·스케일링 생략 조건도 허용한다.
- 거리 기반 SMOTE/NearMiss와 LR/PCA에는 스케일의 영향을 고려한다.
- 기본 스케일러는 `StandardScaler`로 두고, PCA 성분 수는 Train 내부 CV에서 선택한다.
- 원본+추출은 특성 확장 실험이다. PCA는 Target 관련성을 직접 최적화하는 특성 선택이 아니다.
- SMOTE 합성 표본의 이산값·flag·특성 간 제약을 확인한다. 범주형 변수를 유지해야 하면
  해당 변수를 지정한 SMOTENC 등 적절한 방법을 검토한다.
- IR은 리샘플링 전 학습 부분의 음성 수/양성 수로 계산한다.
  LR/DT/ET/RF는 `class_weight`, GB는 `sample_weight`, XGB는 `scale_pos_weight`를 사용한다.
- 우선 가중치와 리샘플링을 동시에 적용하지 않는다. 샘플링 비율과 seed를 기록한다.

### 실험·평가 규칙

1. 원본 특성+보정 없음으로 6개 모델의 기준 성능을 확보한다.
2. 특성 조건과 불균형 보정을 단계적으로 비교하고, 최종 조합의 상호작용도 확인한다.
   단계적 탐색만 수행했다면 전체 조합의 최적성을 주장하지 않는다.
3. 탐색 범위·예산을 기록한다. 튜닝에 사용한 CV 최고 점수는 개발 성능으로 표기한다.
   튜닝된 모델의 CV 일반화 성능을 엄밀히 비교하려면 inner 튜닝/outer 평가의 nested CV를 사용한다.
4. 모델 설정 선택 후 Train OOF 예측으로 threshold를 결정한다.
   OOF는 각 표본을 학습에서 제외한 모델의 예측이다. 이 예측에서 최적화한 F2는 개발 결과다.
5. 기본 threshold 규칙은 F2 최대화다. 대안은 Recall >= 0.95에서 Precision 최대화다.
   두 규칙을 구분하고 주 규칙을 Test 평가 전에 확정한다.
   Recall 기준을 0.95, 0.90 등으로 낮추는 분석은 사전 지정한 민감도 분석으로 보고한다.
6. Threshold 튜닝 효과를 CV로 평가할 때는 평가 fold를 제외한 내부 데이터로 threshold를 정한다.
   검증 Recall 목표가 Test에서도 충족된다는 보장은 없다.
7. 최종 설정을 고정하고 전체 Train에서 파이프라인을 재학습한다. IR도 전체 Train에서 다시 계산한다.
   고정 threshold로 Test를 평가한 뒤 설정을 다시 선택하지 않는다.

| 평가 대상 | 지표 | 해석 |
|---|---|---|
| 순위·판별 | AP, AUROC | 높을수록 좋음, AP와 적분 PR-AUC를 구분 |
| Threshold 결정 | F1, F2, MCC, Precision, Recall | 선택한 threshold를 함께 보고 |
| 확률 예측 오차 | Brier score | 낮을수록 좋음, 분류 Accuracy와 구분 |

- Brier는 보정과 판별 성능 등을 함께 반영한다. 보정 상태는 calibration curve도 함께 확인한다.
- 가중치·리샘플링은 확률 보정에 영향을 줄 수 있다. 확률 보정을 추가하면 Train 내부에서만 학습한다.
- CV 평균·표준편차는 변동성의 기술통계이며, fold를 독립 표본으로 간주하지 않는다.
- 단일 GMSC의 CV fold에 Friedman/Nemenyi를 적용해 CD Diagram을 만들지 않는다.
  여러 독립 데이터셋을 확보하면 적용 조건을 다시 검토한다.
- 사전에 정한 모델 간 Test 지표 차이는 paired bootstrap 신뢰구간으로 분석한다.
  매 반복에서 동일한 Test 행을 두 모델에 공통으로 재추출하며 threshold는 고정한다.
  이는 고정 모델에 대한 Test 표본 불확실성으로, 재학습 변동성 전체를 나타내지는 않는다.
  클래스별 재추출 시 관측 클래스 비율에 조건부인 분석임을 명시한다.
  여러 쌍의 검정을 수행한다면 가설과 다중 비교 보정 방법을 사전에 정한다.

### ass.ipynb 권장 섹션 및 산출물

| 섹션 | 내용 | 저장할 결과 |
|---|---|---|
| 1. 설정·로드 | Target, seed, 지표, 기본 검증 | 설정·데이터 버전·환경 버전 |
| 2. 데이터 분할 | Train/Test 분리·격리 | 분할 인덱스 |
| 3. Train EDA | 분포·왜도·결측·상관관계 | PPT용 EDA 그래프 |
| 4. 파이프라인 | fold 내부 전처리·추출·보정 정의 | 비교 조건 |
| 5. 기준 모델 | 원본+보정 없음의 CV | fold별 지표·OOF 예측 |
| 6. 확장 실험 | 특성·보정 비교·튜닝 | 파라미터·지표·학습 시간 |
| 7. Threshold | 최종 후보의 OOF 기반 결정 | 선택 규칙·threshold |
| 8. 최종 평가 | 전체 Train 재학습·Test 평가 | 지표·PR/ROC·혼동행렬·보정 곡선 |
| 9. 차이 분석·XAI | 신뢰구간·SHAP | 비교 표·설명 그래프 |

- 처음에는 한 노트북으로 시작하고 반복 함수나 긴 학습 코드는 `src/*.py`로 분리한다.
- 커널 재시작 후 위에서 아래로 전체 실행해 재현성을 확인한다.
- 지원되는 트리 모델에는 TreeExplainer를 사용하고, 다른 모델에는 맞는 explainer를 선택한다.
- SHAP 배경 데이터가 필요하면 실제 Train 표본을 사용한다.
- 추출 성분의 SHAP을 원본 변수 기여도로 직접 해석하지 않는다. 원본 특성 모델의 해석도 함께 제시한다.
- SHAP은 인과 효과를 의미하지 않는다.
- PPT 흐름: 데이터 설명 → 분할 → Train EDA → 실험 조건 → CV 비교 → Test 평가 → XAI.
- 전체 데이터 기술통계는 설정 확정 후 설명 목적으로 작성할 수 있다.
  Train EDA와 사용 범위를 구분하고, 그 결과로 모델을 다시 조정하지 않는다.

### 참고 근거

- [Threshold 튜닝과 과적합 방지](https://scikit-learn.org/stable/modules/classification_threshold.html)
- [확률 보정과 Brier score 해석](https://scikit-learn.org/stable/modules/calibration.html)
- [Demšar: 여러 데이터셋에서의 분류기 통계 비교](https://jmlr.org/papers/v7/demsar06a.html)
