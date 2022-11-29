import tarfile
from pathlib import Path

import wget

from opal.conf.conf import OSU_DIR


def download_file(url: Path, fn: Path, overwrite=False):
    """ Downloads a file from the URL to File Name """
    print(f"Downloading from {url} to {fn.as_posix()}")
    if fn.exists() and not overwrite:
        print("File exists, and overwrite is false, skipping")
        return
    wget.download(url, fn.as_posix())


def unzip_tar_bz2(fn: Path):
    """ Unzips the file"""
    tar = tarfile.open(fn.as_posix(), "r:bz2")
    tar.extractall(fn.parent)
    tar.close()


def download_pipeline(url: Path, fn: Path, overwrite=False, cleanup=False):
    """ Downloads the database files

    Args:
        url: https://data.ppy.sh... url
        fn: File Name to download to
        overwrite: Whether to overwrite the tar gz2
        cleanup: Deletes the tar gz2

    """
    download_file(url, fn, overwrite)
    print(f"Unzipping files from {fn}")
    unzip_tar_bz2(fn)
    if cleanup:
        print(f"Cleaning up file {fn}")
        fn.unlink()


url = Path(r"https://data.ppy.sh/2022_10_01_performance_mania_top_1000.tar.bz2")
fn = OSU_DIR / "2022_10_01_performance_mania_top_1000.tar.bz2"
download_file(url, fn)
unzip_tar_bz2(fn)
