#!/usr/bin/env python
"""
Archive training_set에서 소규모 subset을 만들어 빠른 훈련/검증용으로 사용.
사용 예: python create_small_training_set.py 50
        → D:\\archive\\training_set_small (50건) 생성
"""
import os
import shutil
import sys
from pathlib import Path

ARCHIVE = Path(__file__).resolve().parent.parent  # D:\archive
TRAIN = ARCHIVE / "training_set"
OUT = ARCHIVE / "training_set_small"


def main():
    n = 50
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except ValueError:
            pass
    n = max(5, min(n, 780))

    if not (TRAIN / "demographics.csv").exists():
        print("Not found:", TRAIN / "demographics.csv")
        return

    import pandas as pd
    df = pd.read_csv(TRAIN / "demographics.csv")
    cols = ["BidsFolder", "SiteID", "SessionID"]
    if not all(c in df.columns for c in cols):
        print("Missing columns:", cols)
        return
    subset = df.drop_duplicates(subset=cols).head(n)
    if len(subset) < n:
        subset = df.drop_duplicates(subset=cols).iloc[:n]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "physiological_data").mkdir(exist_ok=True)
    (OUT / "algorithmic_annotations").mkdir(exist_ok=True)
    (OUT / "human_annotations").mkdir(exist_ok=True)

    keys = set(zip(subset["BidsFolder"], subset["SessionID"]))
    full_subset = df[df.apply(lambda r: (r["BidsFolder"], r["SessionID"]) in keys, axis=1)]
    full_subset.to_csv(OUT / "demographics.csv", index=False)
    print("Wrote", len(full_subset), "rows to", OUT / "demographics.csv")

    copied_phys = copied_algo = copied_human = 0
    for _, row in subset.iterrows():
        bid, site, ses = row["BidsFolder"], row["SiteID"], int(row["SessionID"])
        for subname, src_dir in [
            ("physiological_data", TRAIN / "physiological_data"),
            ("algorithmic_annotations", TRAIN / "algorithmic_annotations"),
            ("human_annotations", TRAIN / "human_annotations"),
        ]:
            if subname == "algorithmic_annotations":
                fname = f"{bid}_ses-{ses}_caisr_annotations.edf"
            elif subname == "human_annotations":
                fname = f"{bid}_ses-{ses}_expert_annotations.edf"
            else:
                fname = f"{bid}_ses-{ses}.edf"
            src = src_dir / site / fname
            if src.exists():
                dst_dir = OUT / subname / site
                dst_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst_dir / fname)
                if subname == "physiological_data":
                    copied_phys += 1
                elif subname == "algorithmic_annotations":
                    copied_algo += 1
                else:
                    copied_human += 1

    print("Copied EDFs: physiological", copied_phys, "algorithmic", copied_algo, "human", copied_human)
    print("Quick train: python train_model.py -d", str(OUT), "-m model_small -v")


if __name__ == "__main__":
    main()
