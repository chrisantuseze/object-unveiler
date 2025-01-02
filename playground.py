#!/usr/bin/env python3
import zipfile

with zipfile.ZipFile("ppg-dataset-single.zip", 'r') as zip_ref:
    zip_ref.extractall("ppg-dataset2")