"""Download pretraining data from https://huggingface.co/datasets/stas/oscar-en-10k"""
import os
import certifi
from datasets import load_dataset

os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

dataset_name = "stas/oscar-en-10k"
name = dataset_name.split('/')[-1]
ds = load_dataset(dataset_name, split='train')
ds.to_json(f"{name}.jsonl", orient="records", lines=True)
