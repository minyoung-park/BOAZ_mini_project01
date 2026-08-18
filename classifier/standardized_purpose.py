import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./data/all_unique_purpose_labels.csv')

def map_to_standard_category(row):
    if row['has_purpose'] != 'yes':
        return 'None'

    note = str(row.get('purpose_note', '')).lower()
    name = str(row.get('name_key', '')).lower()
    combined = f"{note} {name}"

    # 1) Workout / Fitness
    if any(k in combined for k in ['workout', 'working out', 'work out', 'exercis', 'running', 'runner', 'gym', 'fitness', 'training', 'cardio', 'lifting', 'crossfit', 'jogging', 'weights']):
        return 'Workout'
    # 2) Party / Dance / Social
    elif any(k in combined for k in ['party', 'partying', 'parties', 'dance', 'dancing', 'club', 'celebrat', 'drinking', 'rave', 'pregame', 'pre-game', 'gathering', 'social']):
        return 'Party'
    # 3) Sleep / Relax / Chill
    elif any(k in combined for k in ['sleep', 'sleeping', 'relax', 'relaxing', 'relaxation', 'chill', 'chilling', 'calm', 'calming', 'rest', 'bedtime', 'unwind', 'meditat', 'spa', 'peaceful', 'soothing']):
        return 'Sleep/Relax'
    # 4) Study / Work / Focus
    elif any(k in combined for k in ['study', 'studying', 'focus', 'focusing', 'homework', 'reading', 'concentrat', 'office', 'coding', 'work session', 'for working']):
        return 'Study/Focus'
    # 5) Driving / Travel
    elif any(k in combined for k in ['drive', 'driving', 'road trip', 'roadtrip', 'car', 'cruise', 'cruising', 'commute', 'travel', 'trip']):
        return 'Driving'
    # 6) Holiday / Season Event
    elif any(k in combined for k in ['christmas', 'holiday', 'halloween', 'xmas', 'thanksgiving', 'easter', 'festive']):
        return 'Holiday'
    # 7) Summer / Seasonal
    elif any(k in combined for k in ['summer', 'beach', 'pool', 'vacation', 'sunshine', 'sunny', 'spring', 'fall', 'autumn', 'winter']):
        return 'Seasonal/Summer'
    # 8) Worship / Faith
    elif any(k in combined for k in ['worship', 'praise', 'church', 'christian', 'gospel', 'prayer', 'jesus', 'god', 'hymn', 'faith', 'bible']):
        return 'Worship'
    # 9) Daily Routine
    elif any(k in combined for k in ['shower', 'showering', 'morning', 'wake up', 'breakfast', 'sunrise', 'cooking', 'dinner', 'kitchen', 'baking', 'cleaning', 'clean']):
        return 'DailyRoutine'
    # 10) Romance / Wedding
    elif any(k in combined for k in ['wedding', 'romance', 'romantic', 'date night', 'love', 'intimate']):
        return 'Romance/Wedding'
    # 11) Gaming
    elif any(k in combined for k in ['gaming', 'game', 'gamer', 'video game']):
        return 'Gaming'
    else:
        return 'Other_Purpose'

df['standard_category'] = df.apply(map_to_standard_category, axis=1)

purpose_labels_clean = df[df['has_purpose'] == 'yes'].copy()

output_csv = 'standardized_purpose_labels.csv'
purpose_labels_clean.to_csv(output_csv, index=False, encoding='utf-8-sig')

print(f"대분류 매핑 완료 및 파일 저장: {output_csv}")
print("\n--- 카테고리별 고유 제목 수 및 전체 플레이리스트 커버리지 ---")
summary = purpose_labels_clean.groupby('standard_category').agg(
    unique_titles=('name_key', 'count'),
    total_playlists=('n_playlists', 'sum')
).sort_values(by='total_playlists', ascending=False)
print(summary)

plt.figure(figsize=(10, 6))
bars = plt.barh(summary.index[::-1], summary['total_playlists'][::-1], color='royalblue')

plt.title('Total Playlists per Standard Purpose Category', fontsize=14, pad=15)
plt.xlabel('Number of Playlists', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.6)

for bar in bars:
    w = bar.get_width()
    plt.text(w + 500, bar.get_y() + bar.get_height()/2, f"{int(w):,}", va='center', fontsize=9)

plt.tight_layout()
plt.show()