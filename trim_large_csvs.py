"""
trim_large_csvs.py
Scans all CSVs in the project folder. Any file over 50MB gets
trimmed to a 1000-row sample saved as data/samples/<original_path>.
Original large files are left untouched on disk but added to .gitignore.
"""

import os
import csv
import shutil
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
ROOT_DIR     = Path(__file__).parent          # Run from your NGEA folder
SAMPLE_ROWS  = 1000
SIZE_LIMIT   = 50 * 1024 * 1024              # 50 MB
SAMPLE_DIR   = ROOT_DIR / "data" / "samples"
GITIGNORE    = ROOT_DIR / ".gitignore"
# ─────────────────────────────────────────────────────────────────────────────

def trim_csv(src: Path, dest: Path, n_rows: int = SAMPLE_ROWS):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "r", encoding="utf-8", errors="replace") as fin, \
         open(dest, "w", newline="", encoding="utf-8") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        header = next(reader, None)
        if header:
            writer.writerow(header)
        for i, row in enumerate(reader):
            if i >= n_rows:
                break
            writer.writerow(row)
    return dest

def update_gitignore(large_files: list[Path]):
    existing = GITIGNORE.read_text(encoding="utf-8") if GITIGNORE.exists() else ""
    lines = existing.splitlines()
    added = []
    for f in large_files:
        rel = str(f.relative_to(ROOT_DIR)).replace("\\", "/")
        if rel not in lines:
            lines.append(rel)
            added.append(rel)
    # Make sure samples/ is NOT ignored
    if "!data/samples/" not in lines:
        lines.append("!data/samples/")
    GITIGNORE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return added

def main():
    print("\n📂 Scanning for large CSV files...\n")
    all_csvs = list(ROOT_DIR.rglob("*.csv"))
    large = [f for f in all_csvs if f.stat().st_size > SIZE_LIMIT
             and "data/samples" not in str(f).replace("\\", "/")]

    if not large:
        print("  ✅ No CSV files exceed 50MB — nothing to trim.")
        return

    trimmed = []
    for src in large:
        size_mb = src.stat().st_size / 1e6
        rel     = src.relative_to(ROOT_DIR)
        dest    = SAMPLE_DIR / rel
        print(f"  ✂️  {rel}  ({size_mb:.1f} MB)  →  sample ({SAMPLE_ROWS} rows)")
        trim_csv(src, dest, SAMPLE_ROWS)
        trimmed.append(src)
        print(f"      Saved: data/samples/{rel}")

    added = update_gitignore(trimmed)
    print(f"\n  📝 Added {len(added)} entries to .gitignore")
    print(f"  ✅ Done! {len(trimmed)} files trimmed → data/samples/")
    print("\n  Next steps:")
    print("    git add data/samples/ .gitignore")
    print("    git commit -m \"Add CSV samples and update gitignore\"")
    print("    git push")

if __name__ == "__main__":
    main()
