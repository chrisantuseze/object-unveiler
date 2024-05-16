#!/usr/bin/env python3
import zipfile

with zipfile.ZipFile("pc-ou-dataset3.zip", 'r') as zip_ref:
    zip_ref.extractall("pc-ou-dataset3")

# with zipfile.ZipFile("ppg-dataset.zip", 'r') as zip_ref:
#     zip_ref.extractall("ppg-dataset")