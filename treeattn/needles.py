"""Needle-in-Wikipedia test for ANY language model: real Wikipedia text as filler, a few invented facts hidden in it, and a
completion prompt at the end that asks for one of them.

  fact  : "The secret word of the vault in <Place> is <word>."      (3 facts per haystack, at different depths)
  query : "The secret word of the vault in <Place> is"                (ends the prompt; the answer is the next token)

The answer is a single token chosen at random from rare whole-word tokens, so guessing succeeds about 1 time in
len(answer_ids). Each case also has a CONTROL where the queried fact is missing from the haystack (distractor facts stay):
its accuracy is the chance level a model reaches without retrieval. Everything here is plain Python/numpy: the model is
reached only through a callable logits_fn(ids) -> 1-D array of next-token logits, so the tree model and nanoGPT run the same test."""
import numpy as np

SYL = ['ka', 'lo', 'mi', 'ra', 'ven', 'dor', 'tul', 'sha', 'bri', 'nor', 'el', 'um', 'zan', 'pe', 'ji', 'quo']
PREFIX = '\n\nThe secret word of the vault in %s is'
FREE_PROMPTS = [
    'The history of the Roman Empire',
    'In mathematics, a prime number is',
    'The capital of France is',
    'Albert Einstein was born in',
    'The Amazon River flows through',
    'Question: Who wrote the play Hamlet?\nAnswer:',
    'Photosynthesis is the process by which',
    'The Second World War began when',
]


def answer_ids_from_vocab(vocab, min_id=3000, lo=4, hi=10):
    """Rare whole-word tokens: ' word' style (byte-level BPE marks the space with 'G-dot'), letters only, lo..hi letters."""
    out = []
    for tok, i in vocab.items():
        w = tok[1:] if tok[:1] in ('\u0120', ' ') else None
        if w and w.isalpha() and w.isascii() and lo <= len(w) <= hi and i >= min_id:
            out.append(i)
    return sorted(out)


def make_place(rng):
    return ''.join(rng.choice(SYL, rng.integers(2, 4))).capitalize()


def build_trial(val, enc, answer_ids, L, depth, rng, n_facts=3, with_needle=True):
    """-> (ids, answer_token, info). len(ids) == L. depth in (0,1): where the queried fact starts (fraction of the haystack)."""
    places = []
    while len(places) < n_facts:
        p = make_place(rng)
        if p not in places:
            places.append(p)
    words = [int(w) for w in rng.choice(answer_ids, n_facts, replace=False)]
    prefixes = [PREFIX % p for p in places]
    tail = enc('.\n\n')
    facts = [enc(pf) + [w] + tail for pf, w in zip(prefixes, words)]
    qi = int(rng.integers(n_facts))
    query = enc(prefixes[qi])
    inserted = [(depth if i == qi else None, facts[i]) for i in range(n_facts) if with_needle or i != qi]
    others = [float(rng.uniform(0.05, 0.95)) for _ in inserted]
    fr = []
    for (d, f), o in zip(inserted, others):
        if d is None:
            while abs(o - depth) < 0.05:
                o = float(rng.uniform(0.05, 0.95))
            d = o
        fr.append((d, f))
    hay_len = L - len(query) - sum(len(f) for _, f in fr)
    assert hay_len > 16, 'window too short for the facts'
    if len(val) <= hay_len + 1:                                                # tiny val set (tests): wrap the filler around
        val = np.tile(np.asarray(val), hay_len // max(len(val), 1) + 2)
    s = int(rng.integers(0, len(val) - hay_len - 1))
    hay = [int(v) for v in val[s:s + hay_len]]
    out, prev = [], 0
    for d, f in sorted(fr, key=lambda z: z[0]):
        pos = int(d * hay_len)
        out += hay[prev:pos] + f
        prev = pos
    out += hay[prev:] + query
    return out, words[qi], {'place': places[qi], 'word': words[qi], 'fact_prefix': prefixes[qi], 'facts': list(zip(places, words))}


def run_needle_eval(logits_fn, val, enc, answer_ids, L, depths, n, seed, n_control=None):
    """-> ({depth: accuracy}, control_accuracy). Teacher-forced: correct when argmax of the next-token logits is the hidden word."""
    res = {}
    for d in depths:
        ok = 0
        for k in range(n):
            rng = np.random.default_rng([seed, L, int(d * 1000), k])
            ids, ans, _ = build_trial(val, enc, answer_ids, L, d, rng)
            ok += int(int(np.argmax(logits_fn(ids))) == ans)
        res[d] = ok / n
    nc = n_control or n * 2
    ok = 0
    for k in range(nc):
        rng = np.random.default_rng([seed, L, 999999, k])
        ids, ans, _ = build_trial(val, enc, answer_ids, L, 0.5, rng, with_needle=False)
        ok += int(int(np.argmax(logits_fn(ids))) == ans)
    return res, ok / nc


def needle_demo(logits_fn, gen_fn, val, enc, dec, answer_ids, L=1024, depth=0.5, seed=7):
    """Printable demo of one case: the hidden facts, the question, the expected word, and the model's answer."""
    rng = np.random.default_rng([seed, L])
    ids, ans, info = build_trial(val, enc, answer_ids, L, depth, rng)
    lines = ['  [needle test, %d tokens of Wikipedia with 3 hidden facts; the queried fact sits at ~%d%% depth]' % (L, int(depth * 100))]
    for p, w in info['facts']:
        lines.append('    hidden fact: "The secret word of the vault in %s is%s."' % (p, dec([w])))
    lines.append('    QUESTION (end of the prompt): "%s"' % info['fact_prefix'].strip())
    lines.append('    expected next word:%s' % dec([ans]))
    top = int(np.argmax(logits_fn(ids)))
    lines.append('    model next word:%s   %s' % (dec([top]), 'CORRECT' if top == ans else 'wrong'))
    return '\n'.join(lines)
