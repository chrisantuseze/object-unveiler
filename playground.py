#!/usr/bin/env python3
import zipfile

with zipfile.ZipFile("ppg-dataset23.zip", 'r') as zip_ref:
    zip_ref.extractall("ppg-dataset23")