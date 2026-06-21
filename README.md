# Transformer Problem Solver

This repository contains compact PyTorch examples for sequence-to-sequence style
algorithmic tasks such as reversing integer sequences.

The project intentionally separates three ideas that are easy to conflate:

1. **Non-autoregressive Transformer decoding** reduces sequential model calls.
2. **Single-Pass Index Routing** solves deterministic position-routing tasks with
   vectorized gathers.
3. **LazyRoutedSequence** provides O(1) setup and O(1) random access when only a
   few output positions are needed.

None of these make full answer materialization O(1). If a caller needs to print,
store, or otherwise consume all N output tokens, writing those N tokens is an
unavoidable O(N) lower bound.

## Non-autoregressive Transformer

`TransformerSolver` does not generate answer tokens one by one. It encodes the
source once and predicts all output positions from positional decoder queries in
one forward pass. This primarily improves **latency** by removing N sequential
model invocations.

Important limitations:

- The caller must know or choose `output_length` before decoding.
- Transformer attention is still length-dependent; a single forward pass is not
  O(1). Self-attention/cross-attention costs grow with sequence length.
- Non-autoregressive generation can struggle on open-ended language generation
  because output tokens are less explicitly conditioned on previously generated
  tokens. It is best suited here because reverse/copy tasks have deterministic
  answers.

## Single-Pass Index Routing

For tasks where the answer is a deterministic rearrangement of the input, this
repo includes **Single-Pass Index Routing**. It builds every output-to-input
position mapping at once and then uses one vectorized gather to emit the full
answer. That means answer length only adds the unavoidable O(N) cost of writing N
tokens; it does not add O(N²) repeated autoregressive decoding work.

This is not a general text-generation algorithm. It applies when an output token
can be obtained from an input position via a known mapping, such as reverse,
copy, or other routing/permutation-like tasks.

## Lazy routed view

`LazyRoutedSequence` is a routed view, not a fully materialized answer. It has
O(1) setup, O(1) length lookup, and O(1) random access for one requested output
token. This is useful only when the caller needs a small subset of positions or
wants to defer materialization. If the caller eventually reads every token, the
total work is O(N).

## Fair complexity comparison

| Method | Total work to materialize all N output tokens | Sequential model calls / latency | Single output-token lookup | Applies to |
| --- | --- | --- | --- | --- |
| Autoregressive Transformer | Often O(N) model calls plus growing attention/cache work | N dependent calls | Not independent; previous tokens required | General generation |
| Non-autoregressive Transformer | One length-dependent Transformer forward, not O(1) | 1 model call | Usually computed as part of full pass | Fixed/known-length outputs |
| Single-Pass Index Routing | O(N) gather/write | 0 model calls | O(1) after routing math | Deterministic routing/permutation tasks |
| LazyRoutedSequence | O(N) if fully materialized | 0 model calls | O(1) per requested token | Partial/random access to routing tasks |

## 1B decoder-only Transformer preset

`billion_transformer.py` defines a decoder-only Transformer preset with about
one billion trainable parameters. By default it builds on the `meta` device to
inspect the architecture and parameter count without allocating the full weights:

```bash
python billion_transformer.py
```

To directly materialize the 1B model with real CPU weights, use bfloat16 to keep
parameter memory around 2 GiB:

```bash
python billion_transformer.py --materialize --dtype bfloat16 --device cpu
```

This creates the actual model object and initializes real parameters. It is still
not out-of-the-box training on a small machine: optimizer states, activations,
checkpointing, and data loading require substantially more memory and usually
distributed execution.

## Cloud training entry point

`train_billion_transformer.py` is the training launcher. Use `--model-size small`
for local smoke tests and `--model-size 1b` on cloud GPUs. The default data path
uses synthetic next-token batches so the distributed training stack can be tested
without adding tokenizer or dataset dependencies; replace `sample_batch` with a
real tokenized corpus loader for production-quality training.

Local smoke test:

```bash
python train_billion_transformer.py --model-size small --steps 1 --batch-size 1 --seq-len 8 --device cpu --dtype float32
```

Cloud-style launch example:

```bash
torchrun --nproc_per_node=8 train_billion_transformer.py --model-size 1b --steps 100000 --batch-size 2 --seq-len 2048 --dtype bfloat16 --device cuda
```

Actually training a strong 1B model requires cloud GPU capacity, a large
tokenized dataset, checkpoint storage, monitoring, and many hours or days of
compute. This repository provides the model and training harness; it does not
include cloud credentials or a proprietary dataset.

## Install

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python -m transformer_solver --epochs 3 --train-samples 256 --val-samples 64
```

## Test

```bash
python -m unittest discover -s tests -v
```
