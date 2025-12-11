import os
import urllib.request
import tarfile

# Download the IMDb dataset with .tar.gz file.
def download_file(url: str, save_path: str):
    def progress(bnum, bsize, total_size):
        downloaded = bnum * bsize
        percent = downloaded / total_size * 100
        print(f"\rDownloading: {percent:.2f}% ({downloaded}/{total_size} bytes)", end="")
    print("Start to download data...")
    urllib.request.urlretrieve(url, save_path, progress)
    print("Download finished.")

# Extract the .tar.gz file.
def extract_tar_gz(file_path:str, extract_dir: str):
    print("Extracting")
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)
    print("Extraction completed!")

if __name__ == "__main__" :
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    save_path = "aclImdb_v1.tar.gz"
    extract_dir = "."

    if os.path.exists(save_path):
        print("The file has already existed.")
    else: 
        download_file(url, save_path)
        extract_tar_gz(save_path, extract_dir)

    # Remove temporary files
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"Removed compressed file: {save_path}")
    else:
        print("File not found, nothing to remove.")