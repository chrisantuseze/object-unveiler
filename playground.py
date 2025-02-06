#!/usr/bin/env python3
import zipfile

with zipfile.ZipFile("pc-ou-dataset2-double.zip", 'r') as zip_ref:
    zip_ref.extractall("pc-ou-dataset2-double")