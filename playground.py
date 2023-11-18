#!/usr/bin/env python3
import sys
import zipfile
import pickle
import os

# with zipfile.ZipFile("ou-dataset-consolidated2.zip", 'r') as zip_ref:
#     zip_ref.extractall("ou-dataset-consolidated2")

with zipfile.ZipFile("new.zip", 'r') as zip_ref:
    zip_ref.extractall("new")

