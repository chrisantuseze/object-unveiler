#!/usr/bin/env python3
import zipfile

with zipfile.ZipFile("ppg-dataset2.zip", 'r') as zip_ref:
    zip_ref.extractall("ppg-dataset2")