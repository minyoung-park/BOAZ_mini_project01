import os
import json
import glob
import math
import numpy as np
import pandas as pd
from collections import defaultdict
import kagglehub

label_df = pd.read_csv('standardized_purpose_labels.csv')

valid_labels = label_df[
    (label_df['has_purpose'] == 'yes') &
    (~label_df['standard_category'].isin(['None', 'Other_Purpose']))
]
name_to_cat = valid_labels.set_index('name_key')['standard_category'].to_dict()

print(f"매핑 가능한 고유 플리 제목 수: {len(name_to_cat):,}개")

dataset_dir = kagglehub.dataset_download("himanshuwagh/spotify-million")
data_path = os.path.join(dataset_dir, "data") if os.path.exists(os.path.join(dataset_dir, "data")) else dataset_dir
slice_files = sorted(glob.glob(os.path.join(data_path, "mpd.slice.*.json")))

print(f"탐색할 슬라이스 파일 수: {len(slice_files)}개")

track_cat_counts = defaultdict(lambda: defaultdict(int))
track_meta = {}  
target_slices = slice_files[:100]

for idx, file_path in enumerate(target_slices, 1):
    if idx % 20 == 0 or idx == len(target_slices):
        print(f"[{idx}/{len(target_slices)}] 프로필 집계 중: {os.path.basename(file_path)}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for pl in data['playlists']:
        clean_title = pl.get('name', '').lower().strip()
        category = name_to_cat.get(clean_title)

        if category:
            for tr in pl['tracks']:
                uri = tr['track_uri']
                track_cat_counts[uri][category] += 1
                if uri not in track_meta:
                    track_meta[uri] = (tr['track_name'], tr['artist_name'])

MIN_K = 5
profile_rows = []

for uri, counts in track_cat_counts.items():
    total_occurrences = sum(counts.values())

    if total_occurrences >= MIN_K:
        dist = {cat: cnt / total_occurrences for cat, cnt in counts.items()}

        dominant_cat = max(dist, key=dist.get)
        confidence = dist[dominant_cat]

        entropy = -sum(p * math.log2(p) for p in dist.values() if p > 0)

        t_name, a_name = track_meta.get(uri, ('Unknown', 'Unknown'))

        profile_rows.append({
            'track_uri': uri,
            'track_name': t_name,
            'artist_name': a_name,
            'total_count': total_occurrences,
            'dominant_category': dominant_cat,
            'confidence': round(confidence, 4),
            'entropy': round(entropy, 4),
            'distribution': json.dumps(dist)
        })

track_profiles_df = pd.DataFrame(profile_rows)
output_profile_csv = 'track_category_profiles.csv'
track_profiles_df.to_csv(output_profile_csv, index=False, encoding='utf-8-sig')

print(f"\n트랙 프로필 구축 완료: 총 {len(track_profiles_df):,}개 트랙")
display(track_profiles_df.sort_values(by='total_count', ascending=False).head(10))