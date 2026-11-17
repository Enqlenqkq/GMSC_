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
  `예상질문_답변.txt`다. PPT는 본문 8장·부록 2장이고 발표자 노트를 포함한다.

## 정리 기준

중간 OOF 예측, 전체 threshold 탐색표, 슬라이드 PNG, Python 캐시와 예전 Lab 출력은
다시 생성할 수 있어 삭제하고 `.gitignore`에 등록했다. 최종 Test 예측과 지표,
분할 인덱스, 발표자료, 재현 소스는 보존한다. 중간 CSV는 노트북·스크립트 재실행 시
생성되지만 Git에는 포함하지 않는다.
