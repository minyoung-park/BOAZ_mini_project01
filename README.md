# MPD BPR-MF 베이스라인

Spotify Million Playlist Dataset에서 `playlist = user`, `track_uri = item`으로 간주해 학습하는
PyTorch/CUDA BPR-MF 베이스라인입니다. 데이터셋은 포함되어 있지 않습니다.

## 환경

- Python 3.12
- PyTorch 2.12.1 + CUDA 13.0
- NumPy 2.5.2
- NVIDIA GPU 및 호환 드라이버

PyTorch wheel이 CUDA runtime을 포함하므로 별도의 CUDA Toolkit은 필요하지 않습니다.

## 설치

PowerShell에서 다음 명령을 실행합니다.

```powershell
uv venv .venv --python 3.12
.venv\Scripts\Activate.ps1
uv pip install -r requirements-cuda.txt
```

설치 확인:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 데이터 준비

MPD JSON slice가 들어 있는 폴더를 `--data-dir`로 전달합니다. 폴더는 다음 형태여야 합니다.

```text
외부 데이터 폴더/
├── mpd.slice.0-999.json
├── mpd.slice.1000-1999.json
└── ...
```

## 실행

빠른 확인:

```powershell
python src\bpr_mf_cuda.py `
  --data-dir C:\path\to\mpd\data `
  --max-slices 1 `
  --epochs 1 `
  --samples-per-epoch 10000
```

일부 slice 학습:

```powershell
python src\bpr_mf_cuda.py `
  --data-dir C:\path\to\mpd\data `
  --max-slices 10 `
  --epochs 10 `
  --batch-size 8192
```

전체 MPD 학습 중 VRAM이 부족하면 `--cpu-interactions`를 사용할 수 있습니다.

```powershell
python src\bpr_mf_cuda.py `
  --data-dir C:\path\to\mpd\data `
  --max-slices 0 `
  --epochs 10 `
  --batch-size 8192 `
  --cpu-interactions
```

## 데이터 분할과 평가

각 플레이리스트에서 중복 트랙을 제거한 뒤 마지막에서 두 번째 트랙은 validation, 마지막 트랙은
test로 사용합니다. 나머지 트랙만 BPR 학습에 사용합니다.

평가 지표는 다음과 같습니다.

- `HitRate@K`: 정답 트랙이 상위 K개 안에 포함된 비율
- `NDCG@K`: 정답 트랙의 상위 K개 내 순위 품질
- `sampled_auc`: 정답이 샘플링한 미관찰 트랙보다 높은 점수를 받은 비율

기본적으로 사용자마다 negative 100개를 평가하며 `--eval-negatives`로 변경할 수 있습니다.

## 출력

기본 체크포인트 경로는 `artifacts/bpr_mf_cuda.pt`입니다. 체크포인트에는 다음이 들어 있습니다.

- 사용자 및 아이템 임베딩
- 원본 playlist ID 목록
- 원본 Spotify track URI 목록
- 실행 설정
- validation/test 평가 결과

## 주요 옵션

```text
--factors             임베딩 차원, 기본값 64
--epochs              학습 epoch 수, 기본값 10
--batch-size          학습 미니배치 크기, 기본값 8192
--lr                  학습률, 기본값 0.01
--reg                 L2 정규화 계수, 기본값 0.0025
--eval-negatives      평가 negative 수, 기본값 100
--seed                난수 시드, 기본값 42
--cpu-interactions    학습 상호작용을 CPU에 보관해 VRAM 절약
```
