# GMSC 연체 예측 과제

`ass.ipynb`에서 모델·PCA·불균형 보정·튜닝을 비교하고, Train OOF에서 선택한 임계값을 고정해 Test를 평가한다.

## 실행

1. Python 가상환경에 `requirements.txt`의 패키지를 설치하고 노트북 커널로 선택한다.
2. 작업 디렉터리를 이 저장소 루트로 지정하고 `ass.ipynb`를 위에서부터 실행한다.
3. `givemesomecredit.csv`는 가공된 입력 데이터이며 `split_indices.npz`는 저장한 분할 인덱스다.
   데이터와 행 순서를 바꾸면 기존 인덱스를 재사용하지 않는다.
4. 공통 seed는 33, Stratified Train/Test는 80/20, Train 교차검증은 5-fold다.

## 파일 구성

| 위치 | 내용 |
|---|---|
| `ass.ipynb` | 한국어 설명·주석·실험 결과를 포함한 주 노트북 |
| `gmsc_preprocessing.py`, `export_final_model.py` | 재로드 가능한 전처리기와 최종 joblib 모델 내보내기 |
| `flow.md`, `add.md` | 수업·설계 참고 메모. 미채택 아이디어도 포함하므로 최종 결과는 노트북과 CSV를 기준으로 확인 |
| `lab.ipynb` | 수업 연습 노트북 |
| `gputest.py` | 별도 CUDA 성능 측정 도구. 주 실험의 필수 의존성이 아니며 실행에는 PyTorch가 필요 |
| `baseline_*.csv` | LR/DT/RF/ET/LGB/XGB 기준 비교 |
| `feature_*.csv` | Original(특수값 flag 포함 9개)과 PCA 비교 |
| `balance_*.csv`, `balance_comparison.png` | 보정 없음/IR/SMOTE/NearMiss 비교 요약·fold 결과·그림 |
| `xgb_tuning_results.csv`, `xgb_tuned_oof_metrics.csv` | 최초 튜닝과 OOF 임계값 평가 |
| `final_config.json`, `final_test_*.csv`, `final_test_evaluation.png` | 최초 고정 모델의 설정·Test 예측·평가 |
| `presentation_build/` | 추가 튜닝, Recall 비교 및 PowerPoint 제작을 재현하는 소스 |
| `submission/` | 제출용 PPTX·대본·질의응답·결과 요약·Recall 비교·추가 튜닝 근거 |

## 최종 결과와 해석

- 공통 전처리 후 특성 9개를 사용하는 보정 없는 XGB를 선정했다.
- 최초 튜닝 결과: 트리 250개, 깊이 3, 학습률 0.05.
- Train OOF F2 최대화 임계값: 약 0.100395.
- 최초 Test 결과: AP 0.3488, AUROC 0.8580, Recall 0.6658, Precision 0.2570, F2 0.5051.
- Test 확인 후 수행한 마지막 Train 전용 추가 탐색에서도 기존 설정이 유지됐다. Test는 재평가하지 않았다.
- Recall 비교는 기준 6개 모델과 튜닝 XGB에 공통 임계값 0.5를 적용하고,
  최종 F2 임계값 결과는 별도로 표시한다. 큰 Recall 상승을 튜닝만의 효과로 해석하지 않는다.
- 추가 Test 비교표는 설명용이며 최종 모델 재선정에 사용하지 않는다. MLP는 현재 실험에 포함하지 않았다.

## 추가 실험·발표자료 재현

저장소 루트에서 기존 가상환경의 Python을 사용한다.

```powershell
.\.venv\Scripts\python.exe presentation_build\final_tuning.py
.\.venv\Scripts\python.exe presentation_build\prepare_deck.py
powershell -NoProfile -ExecutionPolicy Bypass -File presentation_build\build_deck.ps1
.\.venv\Scripts\python.exe presentation_build\recall_comparison.py --run
```

- `final_tuning.py`는 기존 분할과 노트북의 전처리·설정을 사용해 Train 전용 추가 탐색 결과를 저장한다.
- PowerPoint 제작에는 Windows와 데스크톱 PowerPoint, `python` 명령이 필요하다.
  기본 텍스트·표를 만든 뒤 표준 라이브러리로 편집 가능한 차트와 내장 워크북을 포함한다.
- `recall_comparison.py --run`은 비교표·그래프를 생성한다. 인자 없이 실행하는 방식은
  최초 노트북 셀 추가용이므로 이미 셀이 존재하는 현재 상태에서는 사용하지 않는다.
- 제출물은 `submission/GMSC_7min.pptx`, `발표대본_7분.txt`, `결과정리_제출안내.txt`,
  `예상질문_답변.txt`다. PPT는 본문 8장·부록 3장(XAI 포함)이고 발표자 노트를 포함한다.

## 정리 기준

중간 OOF 예측, 전체 threshold 탐색표, 슬라이드 PNG, Python 캐시와 예전 Lab 출력은
다시 생성할 수 있어 삭제하고 `.gitignore`에 등록했다. 최종 Test 예측과 지표,
분할 인덱스, 발표자료, 재현 소스는 보존한다. 중간 CSV는 노트북·스크립트 재실행 시
생성되지만 Git에는 포함하지 않는다.

## 최종 모델 내보내기·불러오기

노트북 마지막 셀은 학습된 `tuned_pipe`를 저장한다. 커널에 예전 노트북 클래스가 남아 있어도
학습 상태를 유지한 채 모듈 클래스로 교체해 새 Python 세션에서 로드할 수 있도록 한다.
독립 실행은 `.\.venv\Scripts\python.exe export_final_model.py`로 수행한다.
독립 실행은 고정 설정으로 Train만 재학습하며, 기존 Test 예측을 재현하는지만 확인한다.

저장 파일 `submission/final_model.joblib`에는 파이프라인, F2 임계값, 입력 컬럼과 설정이 들어 있다.
동일한 패키지 환경에서 저장소 루트를 모듈 경로로 사용한다. 다른 위치로 전달할 때는
`gmsc_preprocessing.py`도 함께 제공해야 한다.

```python
import joblib

artifact = joblib.load("submission/final_model.joblib")
pipeline = artifact["pipeline"]
X_input = X_new[artifact["input_columns"]]  # 원본 이름을 단축한 8개 입력 컬럼
positive_col = list(pipeline.classes_).index(artifact["positive_label"])
prob = pipeline.predict_proba(X_input)[:, positive_col]
pred = (prob >= artifact["threshold"]).astype(int)
```

`pipeline.predict()`는 기본 임계값을 사용하므로, 최종 F2 기준 예측에는 저장된 임계값을 명시적으로 적용한다.

## Benchmark 참고 비교

`.\.venv\Scripts\python.exe benchmark\compare_models.py`로 외부 모델과 저장된 최종 모델을
현재 Test 30,000건에서 비교한다. 결과는 `submission/benchmark_comparison/`에 저장한다.
`--inspect` 옵션은 외부 모델의 타입·입력 컬럼·파라미터를 출력한다.

- benchmark에는 저장된 원본 컬럼 8개를, 현재 모델에는 자체 전처리 파이프라인을 적용한다.
  외부 전처리기가 제공되지 않아 benchmark는 원본 수치 입력으로 평가했다.
- 공통 임계값 0.5에서 비교하고, 현재 모델의 Train OOF F2 임계값은 별도 행으로 표시한다.
- benchmark 노트북 추가 확인 후 seed=7 분할을 재구성했다. 현재 CSV로 재학습한 모델과
  저장 모델의 전체 예측·트리 구조가 정확히 일치했으며, 현재 Test의 23,978건이 benchmark
  학습 부분과 겹친다. 기존 30,000건 비교는 독립적인 일반화 비교로 사용하지 않는다.
- `.\.venv\Scripts\python.exe benchmark\audit_notebook.py`는 이 검증과 공통 Test
  6,022건 비교를 재현한다. `노트북_비교분석.txt`와 `split_audit_metrics.csv`를 우선 확인한다.
  공통 부분집합도 사후 분석이므로 새로운 외부 검증이나 유의성 증명이 아니다.

## 최종 모델 XAI

`.\.venv\Scripts\python.exe xai_analysis.py`로 저장된 최종 모델을 재학습 없이 설명한다.
Train 실제 배경 표본 100건과 Test 층화 표본 1,000건을 사용한 interventional TreeSHAP이며,
`submission/xai/`에 전역 중요도·beeswarm·상위 변수 의존성·개별 waterfall·수치·해석 안내를 저장한다.

- 설명은 로그·스케일 변환 후 XGB 입력에 대한 것이며, SHAP 단위는 log-odds다.
- 기여도 합계+기준값이 모델 raw 점수와 일치하고 sigmoid 변환이 예측 확률과 일치하는지 검증한다.
- 고위험·저위험·오탐·미탐 사례는 설명용 선택이며 모집단을 대표하지 않는다. 원 단위 값은 `local_cases.csv`에 별도 저장한다.
- SHAP은 모델 판단의 기여도이며 실제 연체의 인과 원인이나 금융 결정의 정당성을 증명하지 않는다.

## 제출 전 보완 분석

`.\.venv\Scripts\python.exe supplementary_analysis.py`는 기존 분할·기준 모델 설정으로
Train 구간별 연체율, LR/XGB의 Raw8·Clean_NoLog·Clean_Log·Zero_Income_flag ablation,
기준 XGB/LGB/IR의 paired stratified bootstrap 1,000회를 계산한다.
산출물은 `submission/supplementary/`에 저장한다. 노트북 섹션 4와 9.1에 표시 셀이 있다.

최초 Test 확인 후의 탐색적 보완이며 최종 모델·설정·임계값은 변경하지 않았다.
Bootstrap은 고정된 기준 모델 간 Test ΔAP의 표본 불확실성으로, CV 평균·재학습·선택 변동성이나
다중 비교 보정 구간으로 해석하지 않는다. DebtRatio와 소득 원인의 단정은 피하고
현재 관측값과 미확인 가공 규칙을 구분한다. 도메인 상호작용 두 후보는 미수행으로 명시했다.
