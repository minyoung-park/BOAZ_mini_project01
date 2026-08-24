# Spotify 2D Profiling & Hierarchical Bayesian Smoothing Classifier

Spotify **Million Playlist Dataset(MPD)** 를 기반으로, 개별 트랙에 **장르(Genre)** 와 **TPO(Time·Place·Occasion)** 를 동시에 추론하는 2차원 확률 프로필을 생성하고, 여기에 **아티스트(Artist)** 레이어를 결합한 3계층 Feature Store를 구축하는 프로젝트입니다. 최종 산출물은 트랙 단위 [장르 × TPO] 확률 벡터이며, 이를 이용해 조건 기반 추천 필터링을 수행합니다.


## 파일 구조
```text
├── data/                                       
│   ├── sample_50k_flat.parquet                  # 50k 플랫 원본 데이터셋 (334만 행) 
│   └── sample_unseen_20k_flat.parquet           # 20k 미학습 신규 검증 데이터셋
├── results_final_feature_store/                 
│   ├── track_2d_hierarchical_profiles.parquet   # [메인] 461,880개 트랙 2D 피처 스토어
│   ├── album_2d_profiles.parquet                # 앨범 단위 2D 프로필 
│   └── artist_2d_profiles.parquet               # 아티스트 단위 2D 프로필
├── BOAZ_MPD_classifier.ipynb                    # 전 과정 스크립트
└── README.md
``` 

---

## 사용 기술

- **언어/환경**: Python 3, Google Colab (GPU: T4)
- **데이터 처리**: pandas, pyarrow, numpy
- **평가**: scikit-learn (`classification_report`, `f1_score`)
- **시각화**: matplotlib, seaborn, koreanize-matplotlib 
- **데이터 소스**: kagglehub (`himanshuwagh/spotify-million`)

---

## 파이프라인
1. **데이터 전처리 & Title-Split**: 
   * MPD 50k 플리(334만 행)를 스트리밍 파싱 후 8대 표준 장르 시드 추출.
   * MD5 해시 기반 Title-Split(Train 80% / Test 20%)으로 데이터 누수(Data Leakage) 원천 차단.
   
2. **도메인 맞춤 가중치 & 2차원 의사 라벨링**:
   * 장르(Hard Purity, $\gamma=1.0$)와 TPO(Soft Purity, $\gamma=0.3$) 가중치 분기 적용.
   * 5만 개 플리 대상 Soft-Gated Voting 및 L1 정규화로 `[8대 장르] × [11대 TPO]` 2D 확률 장부 생성.
   
3. **미학습 데이터셋(Unseen 20k) 일반화 검증**:
   * 학습에 전혀 쓰이지 않은 연속 슬라이스 20,000개 플리를 대상으로 트랙 적중률 및 역추론 성능 검증.
   
4. **3계층 하향식 베이지안 평활화 (Hierarchical Bayesian Smoothing)**:
   * $\text{Track} \rightarrow \text{Album} \rightarrow \text{Artist}$ 관측치 집계 후 상위 성향을 하위 트랙으로 주입.
   * 4단계 모델 비교(Ablation Study)를 통해 최적 가중치($M_{\text{art}}=5.0, M_{\text{alb}}=0.5$) 도출.
   
5. **최종 피처 스토어 확정 및 추천 서빙**:
   * 결손율 0.00%의 완성형 Parquet 피처 스토어 빌드 및 조건 필터링 API 구축.


---

## 주요 결과

### TPO 신호 결손율 및 희소성
| Before (평활화 전) | After (평활화 후) |
| :---: | :---: |
| <img src="https://github.com/user-attachments/assets/2ad0839d-0a48-41b6-81df-28bf574e08fc" width="100%" /> | <img src="https://github.com/user-attachments/assets/ef583753-3c79-4d44-aee7-2b79fc25ac25" width="100%" /> |

### Genre X TPO 2D 결합 히트맵
<img width="2048" height="980" alt="image" src="https://github.com/user-attachments/assets/801ff955-ba5b-487b-909d-b33ea64c2622" />


### 대표 Edge Case 검증
<img width="832" height="505" alt="image" src="https://github.com/user-attachments/assets/68eb0fc8-0a73-4feb-8251-8e198c0dac16" />

---

## 요구 사항

```bash
pip install pandas numpy pyarrow scikit-learn tqdm matplotlib seaborn koreanize-matplotlib kagglehub
```

MPD 데이터 다운로드 시 Kaggle API 인증(`kagglehub`)이 필요합니다.

---

## 사용 예시

```python
from track_2d_feature_store import Track2DFeatureStore

store = Track2DFeatureStore()
rock_workout_tracks = store.filter_tracks(genre="Rock", tpo="Workout", min_prob=0.4)
print(rock_workout_tracks.head())
```

---

## 참고

- 원본 데이터셋 경로 등은 Colab(`/content/...`) 환경 기준으로 하드코딩되어 있어, 로컬 실행 시 경로 수정이 필요합니다.
- 최적 하이퍼파라미터는 50k Ablation Study(순도/아티스트 가중치 튜닝) 결과를 기준으로 고정되어 있습니다.
