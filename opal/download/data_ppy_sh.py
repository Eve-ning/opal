import wget

from opal.conf.conf import OSU_DIR


def download_file(url, file_name, overwrite=False):
    print(f"Downloading from {url} to {file_name.as_posix()}")
    if file_name.exists() and not overwrite:
        print("File exists, and overwrite is false, skipping")
        return
    wget.download(url, file_name.as_posix())


download_file(
    r"https://data.ppy.sh/2022_10_01_performance_mania_top_1000.tar.bz2",
    OSU_DIR / "2022_10_01_performance_mania_top_1000.tar.bz2"
)
