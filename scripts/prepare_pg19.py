"""PG-19 (books) -> byte-level BPE tokenizer + flat uint16 token files. Needs internet (Kaggle: Settings -> Internet on).
Tries the dataset ids below in order; if none loads, use --local_txt_dir with a folder of .txt books instead.
  python scripts/prepare_pg19.py --out data/pg19 --train_tokens 60000000 --vocab 16384
The tokenizer is trained on the first books only. Both the tree model and the dense baseline use the same files."""
import argparse
import glob
import json
import os
import sys

import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--out', default='data/pg19')
p.add_argument('--train_tokens', type=int, default=60_000_000)
p.add_argument('--val_books', type=int, default=24)
p.add_argument('--tok_books', type=int, default=120, help='books used to train the tokenizer')
p.add_argument('--vocab', type=int, default=16384)
p.add_argument('--local_txt_dir', default='')
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


def books(split):
    if a.local_txt_dir:
        files = sorted(glob.glob(os.path.join(a.local_txt_dir, '**', '*.txt'), recursive=True))
        files = files[:-a.val_books] if split == 'train' else files[-a.val_books:]
        for f in files:
            yield open(f, encoding='utf-8', errors='ignore').read()
        return
    from datasets import load_dataset
    last = None
    for name in ['emozilla/pg19', 'deepmind/pg19']:
        try:
            ds = load_dataset(name, split=split, streaming=True)
            it = iter(ds)
            first = next(it)
            print('dataset', name, 'split', split, 'ok', flush=True)
            yield first['text']
            for ex in it:
                yield ex['text']
            return
        except Exception as e:                                    # try the next id
            last = e
            print('dataset', name, 'failed:', repr(e)[:200], flush=True)
    sys.exit('Could not load PG-19 (%r). Turn Internet on, or pass --local_txt_dir with .txt books.' % (last,))


tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
sample = []
for i, t in enumerate(books('train')):
    sample.append(t)
    if len(sample) >= a.tok_books:
        break
tok.train_from_iterator(sample, trainers.BpeTrainer(vocab_size=a.vocab, special_tokens=['<|endoftext|>'],
                                                   initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
tok.save(os.path.join(a.out, 'tokenizer.json'))
eot = tok.token_to_id('<|endoftext|>')
del sample


def encode_split(split, max_tokens=None, max_books=None):
    chunks, n, nb, buf = [], 0, 0, []

    def flush():
        nonlocal n
        for enc in tok.encode_batch(buf):
            arr = np.array(enc.ids + [eot], dtype=np.uint16)
            chunks.append(arr)
            n += len(arr)
        buf.clear()

    for t in books(split):
        buf.append(t)
        nb += 1
        if len(buf) >= 8:
            flush()
            print('%s: %d books, %.1fM tokens' % (split, nb, n / 1e6), flush=True)
        if (max_tokens and n >= max_tokens) or (max_books and nb >= max_books):
            break
    if buf:
        flush()
    return np.concatenate(chunks)


val = encode_split('validation', max_books=a.val_books) if not a.local_txt_dir else encode_split('validation')
val.tofile(os.path.join(a.out, 'val.bin'))
train = encode_split('train', max_tokens=a.train_tokens)
train.tofile(os.path.join(a.out, 'train.bin'))
json.dump({'vocab': tok.get_vocab_size(), 'eot': eot, 'train_tokens': int(len(train)), 'val_tokens': int(len(val))},
          open(os.path.join(a.out, 'meta.json'), 'w'))
print('done | vocab %d | train %.1fM tokens | val %.2fM tokens' % (tok.get_vocab_size(), len(train) / 1e6, len(val) / 1e6))
