"""Lossless compressed path interface for unchanged DiagnosticTrace hooks."""
import gzip
from pathlib import Path


class CompressedTracePath:
    def __init__(self,path):self.path=Path(path)
    def open(self,mode='r',encoding='utf-8'):
        return gzip.open(self.path,mode+'t' if 'b' not in mode and 't' not in mode else mode,encoding=encoding,compresslevel=4)
    def read_text(self,encoding='utf-8'):
        with gzip.open(self.path,'rt',encoding=encoding) as stream:return stream.read()
    def exists(self):return self.path.exists()
    def __str__(self):return str(self.path)
