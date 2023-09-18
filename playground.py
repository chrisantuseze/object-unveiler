#!/usr/bin/env python3
import sys
import zipfile

with zipfile.ZipFile("object-unveiler-ds.zip", 'r') as zip_ref:
    zip_ref.extractall("object-unveiler-ds")