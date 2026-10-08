"""TinyStories -> byte-level BPE tokenizer + flat uint16 token files (train.bin / val.bin / meta.json / tokenizer.json), the same
format as prepare_wiki.py / prepare_pg19.py, so every phase script reads it with --data_dir.
Needs internet for the dataset (Kaggle: Settings -> Internet on): roneneldan/TinyStories, train split for training + tokenizer, its
validation split for validation. Offline test / custom text: --local_txt_dir with a folder of .txt files (one story per file).
Stories are short (~200 tokens), so training / evaluation windows span several stories (cross_docs), stories are joined by an
end-of-text token.
  python scripts/prepare_tinystories.py --out data/tinystories --train_tokens 50000000 --vocab 4096"""
import argparse
import glob
import json
import os
import sys

import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--out', default='data/tinystories')
p.add_argument('--train_tokens', type=int, default=50_000_000)
p.add_argument('--val_stories', type=int, default=3000)
p.add_argument('--tok_stories', type=int, default=20000, help='stories (from the start of train) that also train the tokenizer')
p.add_argument('--vocab', type=int, default=4096)
p.add_argument('--local_txt_dir', default='')
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


def streams():
    """-> (validation texts list, iterator over training texts)."""
    if a.local_txt_dir:
        files = sorted(glob.glob(os.path.join(a.local_txt_dir, '**', '*.txt'), recursive=True))
        texts = (open(f, encoding='utf-8', errors='ignore').read() for f in files)
        val = [t for _, t in zip(range(a.val_stories), texts)]
        return val, texts
    try:
        from datasets import load_dataset
        val_ds = load_dataset('roneneldan/TinyStories', split='validation', streaming=True)
        val = [ex['text'] for _, ex in zip(range(a.val_stories), val_ds)]
        tr_ds = load_dataset('roneneldan/TinyStories', split='train', streaming=True)
        return val, (ex['text'] for ex in tr_ds)
    except Exception as e:
        sys.exit('Could not load roneneldan/TinyStories (%r). Turn Internet on, or pass --local_txt_dir with .txt files.' % (e,))


val_texts, stream = streams()
tok_texts = [t for _, t in zip(range(a.tok_stories), stream)]
tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
tok.train_from_iterator(val_texts[:500] + tok_texts, trainers.BpeTrainer(
    vocab_size=a.vocab, special_tokens=['<|endoftext|>'], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
tok.save(os.path.join(a.out, 'tokenizer.json'))
eot = tok.token_to_id('<|endoftext|>')


def encode(texts):
    return [np.array(e.ids + [eot], dtype=np.uint16) for e in tok.encode_batch(texts)]


val = np.concatenate(encode(val_texts))
val.tofile(os.path.join(a.out, 'val.bin'))
chunks = encode(tok_texts)
n = sum(len(c) for c in chunks)
del val_texts, tok_texts
buf = []
for t in stream:
    if n >= a.train_tokens:
        break
    buf.append(t)
    if len(buf) >= 2048:
        for c in encode(buf):
            chunks.append(c)
            n += len(c)
        buf = []
        print('train: %.1fM tokens' % (n / 1e6), flush=True)
if buf:
    for c in encode(buf):
        chunks.append(c)
        n += len(c)
train = np.concatenate(chunks)
train.tofile(os.path.join(a.out, 'train.bin'))
json.dump({'vocab': tok.get_vocab_size(), 'eot': eot, 'train_tokens': int(len(train)), 'val_tokens': int(len(val)), 'cross_docs': True},
          open(os.path.join(a.out, 'meta.json'), 'w'))
print('done | vocab %d | train %.1fM tokens | val %.2fM tokens' % (tok.get_vocab_size(), len(train) / 1e6, len(val) / 1e6))
