# BEST-X1 재랭킹 (공유용)

이 브랜치는 **중간점검 추천기만** 담습니다. MPD 원본·분석 노트북·분류기는 없습니다.

```bash
git clone -b rerank --single-branch https://github.com/minyoung-park/BOAZ_mini_project01.git
```

## 파일

| 파일 | 역할 |
|---|---|
| `Best-X1정리.md` | 분석 → 실험 → 숫자. 먼저 이 파일 |
| `src/best_rerank.py` | 중간점검 BEST (X1) |
| `src/context_rerank.py` | B0~P1 ablation |
| `src/typed_candidates.py` | FAN (`--centric-only`) |
| `src/bpr_mf.py` | BPR 베이스. 학습은 다시 하지 않음 |
| `colab_bpr_mf_cuda.ipynb` | Colab. 학습 셀 건너뛰고 BEST / FAN 셀 |
| `bpr_mf_outputs/*_best_rerank_n500.json` | BEST vs B0 |
| `bpr_mf_outputs/*_typed_cand_n500_inj80_centric.json` | FAN (팬형만) |
| `bpr_mf_outputs/*_typed_cand_n500_inj80.json` | TYPED (탐색형까지, 탐색 하락 로그) |
| `bpr_mf_outputs/*_context_rerank_n500_x1p1_on.json` | C1/X1/P1 |

체크포인트 `bpr_mf_slices_10.pt`와 MPD slice는 Drive/로컬에 이미 있는 것을 씁니다. 이 브랜치에는 없습니다.

## 실행

GPU, 학습 없음. Drive에 `src/`와 `.pt`가 있으면 Colab에서 BEST 셀만 실행하면 됩니다.

BEST는 X1입니다. 분류기·목적 라벨을 점수에 넣지 않습니다.
