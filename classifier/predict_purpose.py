import ast
import json
import pandas as pd
from collections import Counter

print("1. 사전 데이터 로드 중...")
label_df = pd.read_csv('all_unique_purpose_labels.csv')
profile_df = pd.read_csv('track_category_profiles.csv')

profile_dict = profile_df.set_index('track_uri').to_dict(orient='index')

no_purpose_titles = set(label_df[label_df['has_purpose'] == 'no']['name_key'].dropna().str.lower())
has_purpose_titles = set(label_df[label_df['has_purpose'] == 'yes']['name_key'].dropna().str.lower())

print(f" - 사전 내 '목적 없음(no)' 고유 제목 수: {len(no_purpose_titles):,}개")

def parse_tracks(track_data):
    if isinstance(track_data, list):
        return track_data
    if pd.isna(track_data):
        return []
    try:
        return json.loads(track_data)
    except Exception:
        try:
            return ast.literal_eval(track_data)
        except Exception:
            return []

def analyze_playlist_coherence(playlist_tracks, conf_threshold=0.5, coherence_threshold=0.6):
    voters = []

    for tr in playlist_tracks:
        uri = tr.get('track_uri')
        if uri in profile_dict:
            prof = profile_dict[uri]
            if prof['confidence'] >= conf_threshold:
                voters.append({
                    'track_name': tr.get('track_name', prof.get('track_name', 'Unknown')),
                    'artist_name': tr.get('artist_name', prof.get('artist_name', 'Unknown')),
                    'voted_category': prof['dominant_category'],
                    'track_conf': prof['confidence']
                })

    total_tracks = len(playlist_tracks)
    voter_count = len(voters)

    if voter_count == 0:
        return {
            'predicted_category': 'Unidentifiable (No Profile)',
            'coherence_score': 0.0,
            'voter_count': 0,
            'voter_ratio': f"0/{total_tracks}",
            'category_votes': {},
            'outlier_tracks': []
        }

    cat_counts = Counter([v['voted_category'] for v in voters])
    dominant_cat, dominant_votes = cat_counts.most_common(1)[0]
    coherence_score = dominant_votes / voter_count
    outliers = [
        f"{v['track_name']} - {v['artist_name']} ({v['voted_category']})"
        for v in voters if v['voted_category'] != dominant_cat
    ]

    final_cat = dominant_cat if coherence_score >= coherence_threshold else 'Personal Mix'

    return {
        'predicted_category': final_cat,
        'coherence_score': round(coherence_score, 3),
        'dominant_category': dominant_cat,
        'voter_count': voter_count,
        'voter_ratio': f"{voter_count}/{total_tracks} ({round(voter_count/total_tracks*100, 1)}%)",
        'category_votes': dict(cat_counts),
        'outlier_tracks': outliers
    }

print("\n2. 플레이리스트 CSV 데이터 로드 중...")
df_playlists = pd.read_csv(
    'spotify_sample_4k.csv',
    engine='python',
    on_bad_lines='skip',
    encoding='utf-8'
)
print(f" - 로드 완료: 총 {len(df_playlists):,}개 플레이리스트")

no_purpose_list = []
for _, row in df_playlists.iterrows():
    title = str(row.get('name', '')).strip()
    title_lower = title.lower()

    if title_lower in no_purpose_titles or title_lower not in has_purpose_titles:
        no_purpose_list.append(row)

df_target = pd.DataFrame(no_purpose_list)
print(f" - 필터링된 '목적 불분명' 대상 플리: {len(df_target):,}개")

print("\n=== 모호한 제목 플레이리스트 역추론 테스트 결과 ===")
tested_count = 0
all_results = []

for _, row in df_target.iterrows():
    raw_name = str(row.get('name', ''))
    tracks = parse_tracks(row.get('tracks', []))

    if len(tracks) < 5:
        continue

    res = analyze_playlist_coherence(tracks)

    if res['voter_count'] >= 3:
        all_results.append({
            'name': raw_name,
            'predicted': res['predicted_category'],
            'coherence': res['coherence_score'],
            'votes': res['category_votes']
        })

        if tested_count < 5:
            print(f"\n▶ [제목: '{raw_name}'] (총 {len(tracks)}곡)")
            print(f"  ├─ 역추론 판정 : {res['predicted_category']}")
            print(f"  ├─ 응집도(Score): {res['coherence_score']}")
            print(f"  ├─ 투표 참여율 : {res['voter_ratio']}")
            print(f"  ├─ 득표 분포   : {res['category_votes']}")
            if res['outlier_tracks']:
                print(f"  └─ 이상치 트랙 : {res['outlier_tracks'][:2]}")
            else:
                print(f"  └─ 이상치 트랙 : 없음 (매우 일관됨)")
            tested_count += 1

if all_results:
    res_summary_df = pd.DataFrame(all_results)
    print("\n--- 전체 역추론 카테고리 분포 ---")
    print(res_summary_df['predicted'].value_counts())