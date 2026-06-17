import io

from datasets import load_dataset, Audio, concatenate_datasets

import torchaudio

import argparse

import csv

def extract_durations(dataset):
    durations = []
    for sample in dataset:
        info = torchaudio.info(io.BytesIO(sample['audio']['bytes']))
        durations.append([sample['id'], info.num_frames/info.sample_rate])
    return durations

def write_durations(durations, path):
    with open(path, 'w') as file:
        writer = csv.writer(file)
        writer.writerows(durations)

def load(dataset_name: str):
    if dataset_name == 'libri':
        ds = load_dataset("mythicinfinity/libritts", "all")

        train_dataset = concatenate_datasets([
            ds["train.clean.100"], ds["train.clean.360"], ds["train.other.500"]
        ]).cast_column("audio", Audio(decode=False))

        val_dataset = concatenate_datasets([
            ds["dev.clean"], ds["dev.other"]
        ]).cast_column("audio", Audio(decode=False))
        test_clean = load_dataset(
            'mythicinfinity/librispeech-pc-44khz-opus', 'clean', split='test'
        ).cast_column("audio", Audio(decode=False))
        test_other = load_dataset(
            'mythicinfinity/librispeech-pc-44khz-opus', 'other', split='test'
        ).cast_column("audio", Audio(decode=False))
    return [train_dataset, val_dataset, test_clean, test_other]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default='durations')
    ap.add_argument('--datasets', default='libri')
    args = ap.parse_args()
    datasets = load(args.datasets)
    labels = ['train', 'val', 'test_clean', 'test_other']
    for label, dataset in zip(labels, datasets):
        durations = extract_durations(dataset)
        path = f'{args.out_dir}/{args.datasets}_{label}.csv'
        write_durations(durations, path)

if __name__ == '__main__':
    main()