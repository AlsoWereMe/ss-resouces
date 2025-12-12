from pathlib import Path
import re
import tarfile
import urllib.request
import pandas as pd


# Download a file with a simple progress indicator.
def download(url: str, dst: Path) -> None:
    def progress(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            downloaded = block_num * block_size
            print(f"\rDownloading: {downloaded} bytes", end="")
            return
        downloaded = block_num * block_size
        percent = min(downloaded / total_size * 100, 100.0)
        print(
            f"\rDownloading: {percent:.2f}% ({downloaded}/{total_size} bytes)", end=""
        )

    print("Start downloading...")
    urllib.request.urlretrieve(url, dst.as_posix(), progress)
    print("\nDownload completed.")


# Extract a .tar.gz archive into the target directory.
def extract_tar_gz(archive: Path, target_dir: Path) -> None:
    print("Extracting...")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(path=target_dir)
    print("Extraction completed.")


# Clean raw IMDb text files by removing <br /> tags in-place.
def clean_imdb_texts(aclimdb_dir: Path) -> None:
    """
    Remove '<br />' from all IMDb .txt files.
    This function modifies the extracted files in-place.
    """
    print("Cleaning IMDb text files (<br /> removal)...")
    for txt_path in aclimdb_dir.rglob("*.txt"):
        text = txt_path.read_text(encoding="utf-8")
        cleaned = text.replace("<br />", " ")
        if cleaned != text:
            txt_path.write_text(cleaned, encoding="utf-8")
    print("Text cleaning completed.")


# Load IMDb split (train/test) into a DataFrame with labels: neg=0, pos=1.
def load_split(aclimdb_dir: Path, split: str) -> pd.DataFrame:
    rows = []
    split_dir = aclimdb_dir / split

    for subdir, label in (("neg", 0), ("pos", 1)):
        for txt_path in (split_dir / subdir).glob("*.txt"):
            text = txt_path.read_text(encoding="utf-8")
            rows.append({"text": text, "label": label})

    return pd.DataFrame(rows)


# Count tokens where each English word is treated as one token.
def count_tokens(text: str) -> int:
    return len(re.findall(r"[A-Za-z]+", text))


# Split a CSV file into short/long subsets using 30th/70th percentiles of token length.
def split_csv_by_token_length(csv_path: Path, out_dir: Path, prefix: str) -> None:
    """
    Short: token_len <= 30th percentile (Q30)
    Long : token_len >= 70th percentile (Q70)

    Notes:
    - The original CSV is kept unchanged.
    - token_len is used only for splitting and is NOT saved.
    """
    df = pd.read_csv(csv_path)

    # Compute token length for each sample (temporary column).
    token_len = df["text"].astype(str).apply(count_tokens)

    # Compute quantile thresholds.
    q30 = token_len.quantile(0.3)
    q70 = token_len.quantile(0.7)

    # Split into short/long subsets.
    short_df = df[token_len <= q30]
    long_df = df[token_len >= q70]

    short_path = out_dir / f"{prefix}_short.csv"
    long_path = out_dir / f"{prefix}_long.csv"
    short_df.to_csv(short_path, index=False, encoding="utf-8")
    long_df.to_csv(long_path, index=False, encoding="utf-8")

    # Print split statistics.
    print(f"[{prefix}] total samples: {len(df)}")
    print(f"[{prefix}] Q30={q30:.1f}, Q70={q70:.1f}")
    print(f"[{prefix}] short samples: {len(short_df)}")
    print(f"[{prefix}] long samples : {len(long_df)}")
    print("-" * 60)


def main() -> None:
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    archive = Path("aclImdb_v1.tar.gz")
    extracted_root = Path("aclImdb")
    out_dir = Path("processed")

    # Download + extract only if the extracted dataset directory does not exist.
    if not extracted_root.exists():
        if not archive.exists():
            download(url, archive)
        extract_tar_gz(archive, Path("."))
        # clean raw text files right after extraction
        clean_imdb_texts(extracted_root)

    # Build CSV files for train/test splits.
    print("Building CSV files...")
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        df = load_split(extracted_root, split)
        df.to_csv(out_dir / f"{split}.csv", index=False, encoding="utf-8")
    print(f"CSV files created under: {out_dir.resolve()}")

    # Split each CSV into short/long by token length (30/70 percentiles).
    print("Splitting CSV files by token length...")
    for split in ("train", "test"):
        split_csv_by_token_length(out_dir / f"{split}.csv", out_dir, split)

    # Remove the downloaded archive to save disk space.
    if archive.exists():
        archive.unlink()
        print(f"Removed archive: {archive}")


if __name__ == "__main__":
    main()
