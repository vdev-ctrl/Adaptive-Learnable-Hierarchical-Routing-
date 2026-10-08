"""English Wikipedia -> byte-level BPE tokenizer + flat uint16 token files (train.bin / val.bin / meta.json / tokenizer.json).
Needs internet (Kaggle: Settings -> Internet on). Tries the dataset ids below in order; if none loads, use --local_txt_dir with
a folder of .txt files instead. Articles are concatenated with an end-of-text token and windows may span articles (cross_docs).
  python scripts/prepare_wiki.py --out data/wiki --train_tokens 70000000 --vocab 16384
First --val_articles articles -> validation; the next --tok_articles also train the tokenizer; everything after is training text."""
import argparse
import glob
import json
import os
import sys

import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--out', default='data/wiki')
p.add_argument('--train_tokens', type=int, default=70_000_000)
p.add_argument('--val_articles', type=int, default=3000)
p.add_argument('--tok_articles', type=int, default=15000)
p.add_argument('--vocab', type=int, default=16384)
p.add_argument('--local_txt_dir', default='')
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


def articles():
    if a.local_txt_dir:
        for f in sorted(glob.glob(os.path.join(a.local_txt_dir, '**', '*.txt'), recursive=True)):
            yield open(f, encoding='utf-8', errors='ignore').read()
        return
    from datasets import load_dataset
    last = None
    for name, cfgname in [('wikimedia/wikipedia', '20231101.en'), ('wikipedia', '20220301.en')]:
        try:
            ds = load_dataset(name, cfgname, split='train', streaming=True)
            it = iter(ds)
            first = next(it)
            print('dataset', name, cfgname, 'ok', flush=True)
            yield first['text']
            for ex in it:
                yield ex['text']
            return
        except Exception as e:                                       # try the next id
            last = e
            print('dataset', name, cfgname, 'failed:', repr(e)[:200], flush=True)
    sys.exit('Could not load Wikipedia (%r). Turn Internet on, or pass --local_txt_dir with .txt files.' % (last,))


stream = articles()
val_texts = [t for _, t in zip(range(a.val_articles), stream)]
tok_texts = [t for _, t in zip(range(a.tok_articles), stream)]
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
    if len(buf) >= 256:
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
