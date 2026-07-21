from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SOURCES = [
    ('libri', Path('datasets/libri/cache_libri_train.uint32'), Path('datasets/libri/cache_libri_train.meta.json')),
    ('mls', Path('datasets/mls/cache_mls_train.uint32'), Path('datasets/mls/cache_mls_train.meta.json')),
]

OUT_DIR = Path('datasets/textaudio')
CHUNK = 20_000


def _load_meta(meta_path: Path) -> dict:
    with open(meta_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _check_geometry(name: str, meta: dict, text_seq_len: int, speaker_seq_len: int, speech_seq_len: int) -> None:
    if meta['cache_format'] != 'packed_multimodal_blocks':
        raise ValueError(f'{name}: unexpected cache_format={meta["cache_format"]!r}')
    got = (meta['text_seq_len'], meta['speaker_seq_len'], meta['speech_seq_len'])
    want = (text_seq_len, speaker_seq_len, speech_seq_len)
    if got != want:
        raise ValueError(
            f'{name}: geometry {got} != expected {want} -- rebuild with matching '
            f'--text_seq_len/--speaker_seq_len/--speech_seq_len.'
        )
    if meta['seq_len_tokens'] != sum(want):
        raise ValueError(f'{name}: seq_len_tokens={meta["seq_len_tokens"]} != {sum(want)}')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--text_seq_len', type=int, default=100)
    ap.add_argument('--speaker_seq_len', type=int, default=32)
    ap.add_argument('--speech_seq_len', type=int, default=500)
    ap.add_argument('--out_dir', type=str, default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_cache = out_dir / 'cache_textaudio_train.uint32'
    out_meta = out_dir / 'cache_textaudio_train.meta.json'

    seq_len = args.text_seq_len + args.speaker_seq_len + args.speech_seq_len

    metas = []
    for name, _, meta_path in SOURCES:
        meta = _load_meta(meta_path)
        _check_geometry(name, meta, args.text_seq_len, args.speaker_seq_len, args.speech_seq_len)
        metas.append(meta)

    ref_meta = metas[0]
    for (name, _, _), meta in zip(SOURCES, metas):
        for key in ('total_vocab', 'pad_token_text', 'pad_token_speech', 'text_offset',
                    'text_vocab', 'speaker_offset', 'speaker_vocab', 'speech_offset', 'speech_vocab'):
            if meta[key] != ref_meta[key]:
                raise ValueError(f'{name}: {key}={meta[key]!r} does not match {SOURCES[0][0]}={ref_meta[key]!r}')

    total_rows = sum(int(m['n_sequences']) for m in metas)
    print(f'[combine] combining {total_rows:,} rows from {[s[0] for s in SOURCES]} -> {out_cache}')

    with open(out_cache, 'wb') as out_f:
        for (name, cache_path, _), meta in zip(SOURCES, metas):
            n = int(meta['n_sequences'])
            mm = np.memmap(cache_path, dtype=np.uint32, mode='r', shape=(n, seq_len))
            for start in range(0, n, CHUNK):
                end = min(start + CHUNK, n)
                out_f.write(np.asarray(mm[start:end]).tobytes())
            print(f'[combine] {name}: wrote {n:,} rows')

    meta_out = {
        'cache_format': 'packed_multimodal_blocks',
        'dtype': 'uint32',
        'hf_path': 'combined:' + '+'.join(m['hf_path'] for m in metas),
        'hf_config': None,
        'hf_split': 'train',
        'split_name': 'textaudio_train',
        'n_sequences': total_rows,
        'seq_len_tokens': seq_len,
        'total_vocab': ref_meta['total_vocab'],
        'pad_token_text': ref_meta['pad_token_text'],
        'pad_token_speech': ref_meta['pad_token_speech'],
        'text_seq_len': args.text_seq_len,
        'text_offset': ref_meta['text_offset'],
        'text_vocab': ref_meta['text_vocab'],
        'speaker_seq_len': args.speaker_seq_len,
        'speaker_offset': ref_meta['speaker_offset'],
        'speaker_vocab': ref_meta['speaker_vocab'],
        'speech_seq_len': args.speech_seq_len,
        'speech_offset': ref_meta['speech_offset'],
        'speech_vocab': ref_meta['speech_vocab'],
        'source_datasets': [s[0] for s in SOURCES],
        'source_n_sequences': {s[0]: int(m['n_sequences']) for s, m in zip(SOURCES, metas)},
    }
    with open(out_meta, 'w', encoding='utf-8') as f:
        json.dump(meta_out, f, indent=2)
    print(f'[combine] wrote {out_meta}')


if __name__ == '__main__':
    main()
