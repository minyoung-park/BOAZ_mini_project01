"""고유 제목 전체에 대해 목적 여부만 LLM 판정 (카테고리 없음).

출력:
  has_purpose: yes/no
  purpose_note: 목적이 있으면 짧은 자유 서술, 없으면 빈 문자열
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = Path(__file__).resolve().parent / "outputs"
DEFAULT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You judge whether a Spotify playlist title expresses a clear USAGE PURPOSE
or listening CONTEXT (when/how/in what situation someone would play it).

has_purpose=yes examples:
- activity/situation: workout, study, driving, party, sleep, cooking, shower
- time/season/event: morning, Friday night, Christmas, wedding, birthday
- clear intended use even if wording is informal (roadtrippin, chill out, pregame)

has_purpose=no examples:
- person names, artist names, song titles
- pure genre/mood labels without situation (rap, indie, vibes, chill alone, fire)
- years alone, random words, emojis-only, vague lists (favorites, mix, songs)

Rules:
- Focus on whether the TITLE itself signals purpose/context, not guessing hidden intent.
- If uncertain but leans situational, prefer yes.
- If has_purpose=yes, write a short purpose_note in English (3-8 words).
- If has_purpose=no, purpose_note must be "".
- Do NOT assign fixed categories.
- Return JSON only.
"""

USER_TEMPLATE = """Classify these playlist titles.
Return JSON:
{{"results":[{{"id":0,"has_purpose":"yes"|"no","purpose_note":"...","confidence":"high"|"mid"|"low"}}]}}

Titles:
{titles_block}
"""


def call_openai(client, model: str, batch: pd.DataFrame) -> list[dict]:
    titles_block = "\n".join(
        f"- id={r.sample_id} | title={json.dumps(str(r.name), ensure_ascii=False)}"
        for r in batch.itertuples(index=False)
    )
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(titles_block=titles_block)},
        ],
    )
    data = json.loads(resp.choices[0].message.content)
    results = data.get("results", data if isinstance(data, list) else [])
    if not isinstance(results, list):
        raise ValueError(f"unexpected results type: {type(results)}")
    cleaned: list[dict] = []
    for item in results:
        if isinstance(item, dict) and "id" in item:
            cleaned.append(item)
        elif isinstance(item, str):
            # 가끔 모델이 문자열을 섞어 반환함 -> 무시하고 재시도 유도
            continue
    if len(cleaned) < max(1, len(batch) // 2):
        raise ValueError(
            f"too few valid result dicts: {len(cleaned)}/{len(batch)} raw={results[:3]!r}"
        )
    return cleaned


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--titles",
        default=str(OUT_DIR / "all_unique_titles.csv"),
    )
    parser.add_argument(
        "--out",
        default=str(OUT_DIR / "all_unique_purpose_labels.csv"),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--limit", type=int, default=None, help="앞에서 N개만 (테스트용)")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY missing in .env")

    from openai import OpenAI

    client = OpenAI()
    df = pd.read_csv(args.titles)
    if args.limit:
        df = df.head(args.limit).copy()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    done: set[int] = set()
    if out_path.exists():
        prev = pd.read_csv(out_path)
        rows = prev.to_dict("records")
        done = set(int(x) for x in prev["sample_id"])
        print(f"resume: {len(done)} already labeled")

    pending = df[~df["sample_id"].isin(done)].reset_index(drop=True)
    print(f"pending: {len(pending)} / {len(df)} | model={args.model}")

    for start in range(0, len(pending), args.batch_size):
        batch = pending.iloc[start : start + args.batch_size]
        print(
            f"labeling {len(done) + start + 1}-{len(done) + start + len(batch)} / {len(df)} ...",
            flush=True,
        )
        for attempt in range(4):
            try:
                results = call_openai(client, args.model, batch)
                break
            except Exception as e:
                wait = 2 ** attempt
                print(f"  retry {attempt + 1}: {e} (sleep {wait}s)")
                time.sleep(wait)
        else:
            raise RuntimeError("batch failed")

        by_id = {int(r["id"]): r for r in results if "id" in r}
        for r in batch.itertuples(index=False):
            pred = by_id.get(int(r.sample_id), {})
            has_purpose = pred.get("has_purpose")
            note = pred.get("purpose_note") or ""
            if has_purpose == "no":
                note = ""
            rows.append(
                {
                    "sample_id": int(r.sample_id),
                    "name_key": r.name_key,
                    "name": r.name,
                    "n_playlists": r.n_playlists,
                    "has_purpose": has_purpose,
                    "purpose_note": note,
                    "confidence": pred.get("confidence"),
                    "model": args.model,
                }
            )
        pd.DataFrame(rows).to_csv(out_path, index=False)
        time.sleep(0.08)

    out = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    out.to_csv(out_path, index=False)
    print("\ndone")
    print(out["has_purpose"].value_counts(dropna=False).to_string())
    yes = out[out["has_purpose"] == "yes"]
    print(f"yes titles: {len(yes)}")
    print(f"yes playlist mass (sum n_playlists): {yes['n_playlists'].sum()}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
