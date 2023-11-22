#!/usr/bin/env python3
import sys
import zipfile
import pickle
import os

with zipfile.ZipFile("ppg-ou-dataset.zip", 'r') as zip_ref:
    zip_ref.extractall("ppg-ou-dataset")

