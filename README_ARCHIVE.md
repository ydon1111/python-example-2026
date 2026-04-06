# Archive 데이터로 챌린지 모델 실행하기

이 문서는 **D:\archive** 의 훈련/보조 데이터를 사용해 PhysioNet Challenge 2026 예제 코드를 실행하는 방법을 정리합니다.

**Docker로 간단히 돌리기 (제출과 동일한 방식)** → **[RUN.md](RUN.md)** 참고.

## 데이터 구조 (챌린지 기대 형식과 동일)

챌린지 코드는 아래 구조를 기대합니다. `D:\archive\training_set` 이 이 구조와 동일합니다.

```
<data_folder>/
  demographics.csv          # SiteID, BidsFolder, SessionID, Age, Sex, Race, Ethnicity, BMI, Cognitive_Impairment 등
  physiological_data/
    <SiteID>/
      <BidsFolder>_ses-<SessionID>.edf
  algorithmic_annotations/
    <SiteID>/
      <BidsFolder>_ses-<SessionID>_caisr_annotations.edf
  human_annotations/        # 훈련 시에만 사용 (검증/테스트에는 없음)
    <SiteID>/
      <BidsFolder>_ses-<SessionID>_expert_annotations.edf
```

- **training_set**: 라벨(Cognitive_Impairment) 있음. 원본 780, expert 780, caisr 766 (일부 세션은 caisr 없음 → 해당 레코드는 스킵 또는 zero padding).
- **supplementary_set**: 라벨 없음. physiological_data만 20건 (I0004, I0007). 추론만 가능.

## 1. 환경 설정

```bash
cd D:\archive\python-example-2026-main
pip install -r requirements.txt
```

## 2. 훈련 (Training)

훈련 데이터: `D:\archive\training_set`

```bash
python train_model.py -d D:\archive\training_set -m model -v
```

- `-d`: 데이터 폴더 (demographics.csv + physiological_data 등이 있는 루트)
- `-m`: 모델 저장 폴더 (예: `model`)
- `-v`: 로그 출력

훈련은 780 레코드 기준으로 시간이 꽤 걸릴 수 있습니다 (레코드당 EDF 로딩·특징 추출 포함).

## 3. 추론 (Run model)

### 3-1. 훈련 데이터 일부를 holdout으로 사용하는 경우

holdout용 데이터를 별도 폴더에 두고, 그 폴더의 `demographics.csv` 에 맞춰 추론합니다.

```bash
mkdir holdout_data holdout_outputs
# holdout_data 에 demographics.csv + physiological_data, algorithmic_annotations 복사 (원하는 일부만)
python run_model.py -d holdout_data -m model -o holdout_outputs -v
```

### 3-2. 보조 세트(supplementary)로 추론

라벨은 없고, 예측 결과만 얻을 때:

```bash
python run_model.py -d D:\archive\supplementary_set -m model -o supplementary_outputs -v
```

출력: `supplementary_outputs/demographics.csv` 에 `Cognitive_Impairment`, `Cognitive_Impairment_Probability` 컬럼이 채워집니다.  
supplementary_set 에는 caisr/human 어노테이션 폴더가 없으므로, 해당 특징은 0으로 채워집니다.

## 4. 평가 (Evaluate)

라벨이 있는 CSV와, 모델 추론 결과가 들어간 CSV를 비교할 때 사용합니다.  
**인자 `-d`, `-o` 는 CSV 파일 경로**입니다 (폴더가 아님).

```bash
python evaluate_model.py -d D:\archive\training_set\demographics.csv -o holdout_outputs\demographics.csv -s scores.txt
```

- `-d`: 정답 라벨이 있는 demographics CSV 경로
- `-o`: 모델 예측이 들어간 demographics CSV 경로 (run_model 출력)
- `-s`: 점수 저장 파일 (선택)

평가 스크립트는 `BDSPPatientID` 기준으로 라벨/예측을 매칭합니다. 동일 환자가 여러 세션으로 여러 행이 있으면 중복 인덱스가 생길 수 있으니, 필요 시 demographics를 환자당 1행으로 정리한 뒤 사용하세요.

## 5. 수정 가능한 파일

- **team_code.py** 만 수정합니다.
  - `train_model()`: 특징 추출, 모델 종류/하이퍼파라미터, 저장 방식
  - `load_model()`, `run_model()`: 로드/추론 로직
- `train_model.py`, `run_model.py`, `helper_code.py` 는 수정하지 않습니다 (챌린지에서 그대로 사용).

## 6. 빠른 검증용 (소규모 훈련)

전체 780건 대신 소수 레코드로만 훈련·실행을 확인하려면:

1. `D:\archive\training_set` 에서 demographics.csv 상위 N행만 복사한 `training_small` 폴더를 만들고,
2. 해당 행에 해당하는 `physiological_data`, `algorithmic_annotations`, `human_annotations` 만 같은 상대 경로로 복사한 뒤,
3. 아래처럼 훈련합니다.

```bash
python train_model.py -d D:\archive\training_small -m model_small -v
```

## 7. 요약

| 목적           | 데이터 폴더                    | 명령 예시 |
|----------------|--------------------------------|-----------|
| 훈련           | `D:\archive\training_set`     | `train_model.py -d D:\archive\training_set -m model -v` |
| 추론 (보조)    | `D:\archive\supplementary_set` | `run_model.py -d D:\archive\supplementary_set -m model -o out -v` |
| 평가           | 라벨 CSV / 예측 CSV 파일 경로 | `evaluate_model.py -d <labels.csv> -o <preds.csv> -s scores.txt` |

데이터 경로는 실제 사용하는 드라이브/경로에 맞게 바꿔 사용하면 됩니다.
