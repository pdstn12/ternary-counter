# Fine-Tuning Ternary LLMs Without Master Weights

**Int8 counters keep models ternary with sparse, low-forgetting updates.**

Code and notebooks for the paper *Fine-Tuning Ternary LLMs Without Master Weights: Int8 Counters Keep Models Ternary with Sparse, Low-Forgetting Updates* (Bacem Ben Hamed, 2026). arXiv link: *coming soon*.

## The idea

Fine-tuning a ternary LLM (weights in {−1, 0, +1}) with quantization-aware training keeps an fp32 master copy of every weight plus AdamW moments: about **16 bytes of training state per weight**. Here, every ternary weight gets one **int8 counter** instead:

```
vote  s = -v * G / rms(G)                       # G: the layer's weight gradient
c    <- clip(c + floor(s + u), -127, 127)       # u ~ U[0,1): stochastic rounding, per weight
if |c| >= tau:  t <- clip(t + sign(c), -1, 1);  c <- 0
```

That is **2 bytes per weight** (1.25 with 2-bit packing), with no master weights and no optimizer moments. The model stays ternary throughout, and each layer's counters are updated inside its own backward pass. The paper's setting is τ = 32, v = 0.1, plus a small stochastic leak (0.003) that pulls counters back toward zero.

## Main results

| | result |
|---|---|
| Task accuracy (TriLM 560M: RTE, AG News; BitNet b1.58 2B: Tunisian sentiment, code) | within seed noise of stochastic rounding and Low-Rank Ternary Adaptation (3 seeds) |
| Weights changed | **0.004–0.008%**, 70–1,900× fewer than those methods; the change ships as a < 0.5 MB patch |
| Matched sparse controls (TriLM, RTE) | at the same number of changed weights, stochastic rounding (50.9) and one-shot top-k (47.3) learn nothing; **Bop matches the counter** (63.1 vs 62.6) |
| Forgetting | TriLM: WikiText-2 +12% vs +59% for stochastic rounding; BitNet: no advantage under matched selection |
| Memory (BitNet 2B) | 6.5 GB peak; QAT with bf16 + 8-bit AdamW needs 15.7 GB, and fp32 QAT does not fit in 40 GB |
| **9B** (Ternary Bonsai 2) | full fine-tuning on one A100 at 23.8 GB; 19 minutes on ToolACE raised BFCL from 83.3% to 85.5% (p = 0.006, thinking off) with 0.006% of the weights changed, but the model lost its thinking mode |

The accumulate-and-threshold rule (shared with Bop) is what learns at this sparsity. The counter's specific contribution is doing it with **integer state**. See the paper's Limitations section (Table 4) for everything that did not work.

## Repository layout

```
src/
  bitnet_cpu_finetune.py   fine-tuning of BitNet / TriLM: counter, Bop, SR, top-k, QAT, LoRA, Low-Rank Ternary Adaptation
  ternary_pretrain.py      training small ternary LMs from scratch (Appendix D)
  bonsai_counter.py        PyTorch port of Ternary Bonsai 2 (Qwen3.5 hybrid, PQ2_0 GGUF): counter training + exact GGUF export
  make_tiny_bonsai.py      tiny random model in the same format, to check the port against llama.cpp
  bfcl_lite.py             BFCL evaluation against llama-server, scored with the official AST checker
  BONSAI_PQ2_NOTES.md      notes on the PQ2_0 format and the Hadamard folding
notebooks/
  main/                    tuning, the 3-seed forgetting runs, the matched controls, the mechanism checks
  ablations_and_baselines/ QAT, Low-Rank Ternary Adaptation, code, AG News, GSM8K, ternary gradients
  from_scratch/            51M and 124M from-scratch runs, adaptive tau
  bonsai_9b/               9B fine-tuning (agent data, ToolACE) and the BFCL evaluation
  history/                 earlier exploratory phases, kept for transparency
figures/                   scripts that draw the paper's figures
```

Every notebook installs its own dependencies, embeds the script it runs, and states its expected runtime and its decision rules at the top. They were run on Kaggle (2×T4) and Colab (T4, L4, A100 40 GB).

## Quick start

```bash
pip install -r requirements.txt
cd src
# TriLM 560M on RTE with the paper's counter setting (~15 min on one T4)
python bitnet_cpu_finetune.py --model SpectraSuite/TriLM_560M_Unpacked --load_dtype fp32 --act_quant 0 \
    --arm counter --vote_scale 0.1 --tau 32 --leak 0.003 --fp16_matmul 1 \
    --task rte --train_n 1000 --epochs 2 --tune_n 300 --also_eval wikitext,sst2 --seed 0 --out runs/counter_rte_s0
```

Replace `--arm counter ...` with `--arm bop --bop_gamma 0.003 --bop_thr 1.0` or `--arm sr --sr_lr ...` to run the baselines.

## Licences and data

- Code: MIT (see `LICENSE`).
- `bonsai_counter.py` re-implements the Ternary Bonsai 2 graph from PrismML's llama.cpp fork (MIT). The BFCL checker is loaded at run time from the gorilla repository (Apache-2.0) and is not copied here.
- No model weights or datasets are included. Check each dataset's licence before reusing results: ToolACE (Apache-2.0); UltraData-SFT-Agent (research only, does not grant redistribution or training rights); the Tunisian dialect corpus (unclear provenance).

## Citation

```bibtex
@misc{benhamed2026ternarycounter,
  title  = {Fine-Tuning Ternary LLMs Without Master Weights: Int8 Counters Keep Models Ternary with Sparse, Low-Forgetting Updates},
  author = {Ben Hamed, Bacem},
  year   = {2026},
  note   = {arXiv preprint, to appear}
}
```
