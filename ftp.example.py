"""Copy to ftp.py for local use; keep the filled-in file out of Git."""
from ftplib import FTP

HOST = "xxx"
REMOTE_DIR = "/faces"


def connect_example():
    ftp = FTP()
    ftp.connect(HOST, 21, timeout=30)
    ftp.login("xxx", "xxx")
    ftp.set_pasv(True)
    return ftp
