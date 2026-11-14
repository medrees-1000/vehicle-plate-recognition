"""UC3M-LP (Spanish plates) data plumbing. Test set only: never train on it, never share converted copies.

CLI (skips if the Drive copy already has the right MD5):
    python -m alpr.uc3m download LOCAL_ZIP DRIVE_ZIP
"""
import argparse
import shutil
from pathlib import Path

from alpr import ccpd

URL = "https://zenodo.org/records/17152029/files/UC3M-LP.zip?download=1"
MD5 = "496e3b5e66b78d231f71d85c0e5e11a3"


def fetch(local: str | Path, drive: str | Path, url: str = URL, md5: str = MD5) -> None:
    """Skip if `drive` already matches `md5`; else download to `local` (MD5 checked, raises on mismatch),
    copy to `drive` via .part, check the copy's MD5 too. A Drive file with a wrong MD5 is replaced."""
    drive = Path(drive)
    if drive.exists():
        got = ccpd.file_md5(drive)
        if got == md5:
            print(f"{drive}: MD5 ok, skipping")
            return
        print(f"{drive}: MD5 {got} differs from {md5}, it will be replaced")
    ccpd.download(url, local, md5)
    part = drive.with_name(drive.name + ".part")
    shutil.copyfile(local, part)
    got = ccpd.file_md5(part)
    if got != md5:
        part.unlink()
        raise ValueError(f"MD5 mismatch after copy to {drive}: got {got}, expected {md5}")
    part.replace(drive)
    print(f"{drive}: MD5 ok ({md5})")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="alpr.uc3m")
    d = p.add_subparsers(dest="cmd", required=True).add_parser("download")
    d.add_argument("local")
    d.add_argument("drive")
    a = p.parse_args(argv)
    fetch(a.local, a.drive)


if __name__ == "__main__":
    main()
