# Docker로 간단히 돌리기 (PhysioNet 제출과 동일한 방식)

아래는 **스크립트 없이** 복사해서 쓸 수 있는 명령입니다.  
PhysioNet 제출 시에도 컨테이너 안에서는 `training_data`, `holdout_data`, `model`, `holdout_outputs` 이름을 쓰므로, 같은 방식으로 확인할 수 있습니다.

---

## 1. 폴더 준비 (최초 1회)

`D:\archive` 아래에 출력용 폴더만 만듭니다. 데이터는 이미 있음.

```powershell
mkdir D:\archive\model
mkdir D:\archive\holdout_outputs
```

(훈련 데이터: `training_set`, holdout용: `supplementary_set` 또는 훈련 데이터 일부)

---

## 2. Docker 이미지 빌드

```powershell
cd D:\archive\python-example-2026-main
docker build -t image .
```

---

## 3. 컨테이너 들어가서 훈련 → 추론 → 평가

한 번에 볼륨 마운트해서 bash로 들어갑니다.  
**경로는 Windows 기준 `D:\archive` 이고, 필요하면 본인 경로로 바꾸면 됩니다.**

```powershell
docker run -it -v D:\archive\model:/challenge/model -v D:\archive\supplementary_set:/challenge/holdout_data -v D:\archive\holdout_outputs:/challenge/holdout_outputs -v D:\archive\training_set:/challenge/training_data image bash
```

컨테이너 안에서 아래 **세 줄만** 순서대로 실행합니다.

```bash
python train_model.py -d training_data -m model -v
python run_model.py -d holdout_data -m model -o holdout_outputs -v
python evaluate_model.py -d holdout_data/demographics.csv -o holdout_outputs/demographics.csv -s holdout_outputs/scores.txt
```

끝나면 `exit` 로 나옵니다.

**참고:** `supplementary_set` 에는 `Cognitive_Impairment` 라벨이 없어서 `evaluate_model.py` 가 에러를 낼 수 있습니다. 그럴 때는 평가 줄만 건너뛰거나, holdout으로 훈련 데이터 일부를 복사해 쓰면 됩니다. **제출 시**에는 PhysioNet이 라벨 있는 데이터로 평가합니다.

- 훈련 결과: 호스트 `D:\archive\model`
- 추론 결과: `D:\archive\holdout_outputs\demographics.csv`
- 점수: 호스트 `D:\archive\holdout_outputs\scores.txt`

---

## 4. 한 줄 요약 (컨테이너 안 경로만)

| 단계   | 컨테이너 안에서 실행할 명령 |
|--------|-----------------------------|
| 훈련   | `python train_model.py -d training_data -m model -v` |
| 추론   | `python run_model.py -d holdout_data -m model -o holdout_outputs -v` |
| 평가   | `python evaluate_model.py -d holdout_data/demographics.csv -o holdout_outputs/demographics.csv -s holdout_outputs/scores.txt` |

제출 시에도 PhysioNet이 `training_data`, `holdout_data`, `model`, `holdout_outputs` 를 마운트해서 같은 명령으로 돌리므로, 위와 같이만 맞추면 됩니다.

---

## 5. 빠르게 확인만 할 때 (소규모 훈련)

전체 780건 말고 소수만 쓰려면, 먼저 로컬에서:

```powershell
cd D:\archive\python-example-2026-main
python create_small_training_set.py 50
```

이후 Docker 들어갈 때 `training_set` 대신 `training_set_small` 을 마운트:

```powershell
docker run -it -v D:\archive\model_small:/challenge/model -v D:\archive\supplementary_set:/challenge/holdout_data -v D:\archive\holdout_outputs:/challenge/holdout_outputs -v D:\archive\training_set_small:/challenge/training_data image bash
```

컨테이너 안에서는 동일하게:

```bash
python train_model.py -d training_data -m model -v
python run_model.py -d holdout_data -m model -o holdout_outputs -v
```

(supplementary_set 에는 라벨이 없어서 evaluate 는 생략하거나, holdout 으로 훈련 데이터 일부를 쓰면 평가 가능)
