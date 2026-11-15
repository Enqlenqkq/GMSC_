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
-
