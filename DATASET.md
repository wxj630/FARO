# Dataset: FineWeb (GPT-2 tokenized, `uint16` shards)

This document describes the data used to train and evaluate FARO. The code release
does **not** ship the token files: they are ~1.9 GB of binary shards and are easy to
regenerate or download. Follow the steps below to obtain them.

## 1. What the data is

- **Corpus.** [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb), the
  English `sample-10BT` split: a 10-billion-token sample of web text.
- **License.** FineWeb is released under **ODC-By 1.0** (Open Data Commons
  Attribution License). If you use this data, cite FineWeb:

  ```bibtex
  @article{penedo2024fineweb,
    title={The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale},
    author={Penedo, Guilherme and Kydl{\'\i}{\v{c}}ek, Hynek and Lozhkov, Anton and
            Mitchell, Margaret and Raffel, Colin and Von Werra, Leandro and Wolf, Thomas
            and others},
    journal={arXiv preprint arXiv:2406.17557},
    year={2024}
  }
  ```
- **Tokenization.** GPT-2 BPE via `tiktoken` (`get_encoding("gpt2")`). Documents are
  separated by a single `<|endoftext|>` token, exactly as in the reference trainer
  ([modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)).

## 2. On-disk format

Each shard is a flat binary file read by the loader in `train_gpt.py`:

| Bytes | Contents |
|---|---|
| `0 .. 1023` | Header: `256` × `int32`. `[0]` = magic `20240520`, `[1]` = version `1`, `[2]` = number of tokens that follow. |
| `1024 ..` | Token stream: `uint16` (2 bytes per token), little-endian. |

So a shard with `N` tokens is `1024 + 2N` bytes. Each shard holds `10^8` tokens
(the last one may be shorter). The writer is `data/fineweb.py::write_datafile`.

## 3. How to obtain the shards

Both options write into `data/fineweb10B/`.

### Option A — download the pre-tokenized shards (recommended)

The GPT-2 tokenization of FineWeb-10B is hosted on the Hub, which saves an hour of
CPU tokenization:

```bash
pip install huggingface_hub
python data/cached_fineweb10B.py 9      # 1 val shard + 9 train shards (~1 GB)
```

`cached_fineweb10B.py` downloads from `kjj0/fineweb10B-gpt2`; the integer argument
is the number of train chunks to fetch (default 103 for the full 10 B). **Our runs
used 9 train shards + 1 validation shard.**

### Option B — re-tokenize from the raw corpus

```bash
pip install datasets tiktoken
python data/fineweb.py --version 10B
```

This streams `HuggingFaceFW/fineweb` (`sample-10BT`), tokenizes with GPT-2, and
writes `fineweb_val_000000.bin` followed by `fineweb_train_000001.bin`, `…`.

## 4. Files and splits

```
data/fineweb10B/
├── fineweb_val_000000.bin        # 1 validation shard (~1.0e8 tokens)
├── fineweb_train_000001.bin      # 9 training shards (~9.0e8 tokens total)
├── ...
└── fineweb_train_000009.bin
```

Paths are fixed in `train_gpt.py` (`data/fineweb10B/fineweb_train_*.bin` and
`.../fineweb_val_*.bin`). Two disjoint slices are taken from the validation shard so
that no reported number is computed on probe tokens:

| Slice | Tokens | Used for |
|---|---|---|
| Validation prefix | first `10,485,760` tokens (fixed) | validation loss (`val_tokens`) |
| Probe | next `1,048,576` tokens (`FARO_PROBE_TOKENS`) | FARO routing probe (Eq. 1) and the per-block risk signal |

The train/validation split and the data order are fixed; every matched
prior/ERC pair consumes the identical stream, so a pair differs only in the budget
controller. No test split is used.

## 5. Training regimes

**Clean FineWeb** — the default protocol: the shards above are consumed as-is.

**Scatter (corrupted) — boundary study only.** Implemented in `FARO/regime.py`. A
deterministic hash of `(seed, microbatch index)` marks roughly half of all
micro-batches "dirty"; in a dirty microbatch a fixed fraction of tokens is replaced,
at random positions, with random token ids below the `<|endoftext|>` id (so the
special token is never spuriously introduced). Because dirtiness is keyed on a
per-micro-batch hash, difficulty is **not** temporally clustered — this stresses the
budget controller under noisy per-block risk. In the paper the replacement fraction
is **60%**:

```bash
FARO_REGIME_SCATTER=1 FARO_REGIME_CORRUPTION=0.6 ... python train_gpt.py
```

(The clean protocol leaves both unset.) The regime is reported in the paper as the
boundary of the method, not as a second win.

## 6. Reproducing a run

```bash
python data/cached_fineweb10B.py 9                 # 1. fetch data
pip install -r requirements.txt                    # 2. install deps
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --upgrade

CUDA_VISIBLE_DEVICES=0 FARO_VARIANT=faro FARO_SEED=0 FARO_RUN_ID=test \
  FARO_MAX_STEPS=1285 FARO_MATRIX_LR=0.0015 FARO_MUON_LR=0.023 \
  torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Environment variables are parsed in `FARO/experiment.py`; the queue scripts under
`FARO/queue_*.sh` show the exact settings used for each reported arm (adaptive,
sparse, coordinate-only, Kelly, …).

## 7. Attribution

FineWeb is a third-party dataset. We redistribute no token data; the instructions
above fetch it from the original sources. Cite Penedo et al. (2024) for the corpus
and the modded-nanogpt project for the packing/format conventions.
