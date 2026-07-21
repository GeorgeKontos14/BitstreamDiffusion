from collections import defaultdict

import io

from datasets import load_dataset, Audio, concatenate_datasets

import torchaudio

import argparse

import csv

from tqdm import tqdm

# -----------------------------------------------------------------------------
# Load: LibriTTS.train.all, LibriTTS.dev.clean, LibriSpeechPC.test.clean+.test.other
# -----------------------------------------------------------------------------

def load():
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

# -----------------------------------------------------------------------------
# Compute and Extract durations for each dataset
# -----------------------------------------------------------------------------

def extract_durations(dataset):
    """
    Calculates the duration of each sample for Libri- datasets
    """
    durations = []
    for sample in tqdm(dataset):
        info = torchaudio.info(io.BytesIO(sample['audio']['bytes']))
        durations.append([sample['id'], info.num_frames/info.sample_rate])
    return durations

def write_durations(durations, path):
    """
    Writes (id, seconds) in a csv
    """
    with open(path, 'w') as file:
        writer = csv.writer(file)
        writer.writerows(durations)

# -----------------------------------------------------------------------------
# Hold out one sample per speaker for TTS evaluation
# -----------------------------------------------------------------------------

def select_holdout(dataset, durations_path, out_path, MIN_DURATION=3.0):
    """
    For every speaker on the dataset, hold out one sample to extract speaker
    features for text-to-speech evaluation. The held out samples are selected
    to be at least and as close as possible to MIN_DURATION=3 seconds, in order
    to perform zero-shot text-to-speech.
    """
    rows_per_speaker = defaultdict(list)
    for row in tqdm(dataset):
        rows_per_speaker[row['speaker_id']].append(row['id'])

    durations = {}
    with open(durations_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) < 2:
                continue
            sample_id, duration_str = row[0].strip(), row[1].strip()
            durations[sample_id] = float(duration_str)

    held_out_per_speaker = []
    for spk, ids in rows_per_speaker.items():
        above = [(id, durations[id]) for id in ids if durations[id] >= MIN_DURATION]
        if above:
            held_id, held_dur = min(above, key=lambda x: x[1])
        else:
            held_id, held_dur = max(((id, durations[id]) for id in ids), key=lambda x: x[1])

        print(f'Held-out sample for {spk}: {held_id} ({held_dur:.2f}s)')
        held_out_per_speaker.append((spk, held_id))

    with open(out_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerows(held_out_per_speaker)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default='assets/durations')
    args = ap.parse_args()
    datasets = load()
    labels = ['train', 'val', 'test_clean', 'test_other']
    for label, dataset in zip(labels, datasets):
        print(f'Extracting durations from {label}')
        durations = extract_durations(dataset)
        path = f'{args.out_dir}/{label}.csv'
        write_durations(durations, path)
        
        # Select held out sample for TTS test sets (val + test-clean)
        if label in {'val', 'test_clean'}:
            select_holdout(dataset, f'{args.out_dir}/{label}.csv', f'assets/heldout/{label}.csv')


if __name__ == '__main__':
    main()