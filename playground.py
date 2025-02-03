#!/usr/bin/env python3
import zipfile

with zipfile.ZipFile("ppg-ou-dataset.zip", 'r') as zip_ref:
    zip_ref.extractall("ppg-ou-dataset2")