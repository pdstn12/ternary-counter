#!/usr/bin/env python
"""Fine-tune a ternary LLM (BitNet b1.58 2B4T, TriLM) while keeping it ternary, and compare update rules.

Arms (--arm)
  zeroshot      no training: the released ternary model
  counter       ours: an int8 counter per ternary weight accumulates normalized, stochastically rounded gradient
                votes; the weight moves one step when |counter| >= tau, then the counter resets (2 bytes of state per
                weight, updated inside the backward pass, no master weights)
  bop           Bop baseline: a 16-bit moving average of the same normalized gradient, flip when it crosses a threshold
  sr            stochastic rounding onto the ternary grid, no master weights
  topk          control: sum the counter's votes over the run without changing anything, then move the top-k weights once
  qat           quantization-aware training: latent weights + AdamW (fp32, or bf16 + 8-bit AdamW), straight-through rounding
  lora          LoRA adapters on the frozen ternary base; scored with the adapter and after merging into ternary weights
  tadapt        Low-Rank Ternary Adaptation (re-implementation): a ternary Kronecker mask merged exactly into the weights
  latent_sgd    fp32 master weights with the counter's normalized step (reference)
  latent_adamw  only prints the memory full AdamW fine-tuning would need

Tasks (--task): rte, sst2, agnews, tunisian, code (Magicoder -> EvalPlus), gsm8k, text (your own files)
Forgetting (--also_eval): wikitext, sst2, rte, arc_e, arc_c, hellaswag, obqa, piqa

Example (the paper's counter setting on TriLM 560M / RTE):
  pip install -U torch transformers datasets accelerate psutil
  python bitnet_cpu_finetune.py --model SpectraSuite/TriLM_560M_Unpacked --load_dtype fp32 --act_quant 0 \
      --arm counter --vote_scale 0.1 --tau 32 --leak 0.003 --fp16_matmul 1 \
      --task rte --train_n 1000 --epochs 2 --tune_n 300 --also_eval wikitext,sst2 --seed 0 --out runs/counter_rte_s0
Bop and top-k controls:  --arm bop --bop_gamma 0.003 --bop_thr 1.0   |   --arm topk --topk_frac 3.7e-5
The changed weights are saved as runs/.../flip_patch.npz (positions and new values per layer).
Start with --train_n 64 --eval_n 64 to check that everything runs.
"""
import argparse, json, math, os, random, re, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

TARGETS = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')
TASKS = {
    'sst2':   dict(labels=[' negative', ' positive']),
    'rte':    dict(labels=[' Yes', ' No']),
    'agnews': dict(labels=[' World', ' Sports', ' Business', ' Science']),
    'tunisian': dict(labels=[' negative', ' positive']),
}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', required=True, choices=['zeroshot', 'counter', 'bop', 'sr', 'topk', 'qat', 'lora', 'tadapt', 'latent_sgd', 'latent_adamw'])
    p.add_argument('--model', default='microsoft/bitnet-b1.58-2B-4T-bf16')
    p.add_argument('--out', required=True)
    p.add_argument('--task', default='rte', choices=['sst2', 'rte', 'agnews', 'tunisian', 'text', 'code', 'gsm8k'])
    p.add_argument('--code_dev', type=int, default=200, help='code: held-out training examples for monitoring (never the benchmark)')
    p.add_argument('--code_eval', default='humaneval,mbpp', help='code: EvalPlus benchmarks to run at the end')
    p.add_argument('--code_eval_n', type=int, default=0, help='code: only generate for the first N problems, no scoring (quick checks)')
    p.add_argument('--gsm_n', type=int, default=500, help='gsm8k: score the first N test problems (fixed order)')
    p.add_argument('--gsm_train_file', default='', help='gsm8k: train on these solutions (jsonl question/answer) instead of the '
                   'official human-written ones, e.g. the model\'s own correct answers (self-distillation)')
    p.add_argument('--gsm_selfgen', type=int, default=0, help='gsm8k + zeroshot arm: answer this many training problems with the '
                   'model, keep the correct ones, write them to --gsm_selfgen_out, and stop')
    p.add_argument('--gsm_selfgen_out', default='')
    p.add_argument('--gsm_samples', type=int, default=1, help='selfgen: tries per problem (1 = greedy; >1 = sampling, '
                   'rejection-sampling fine-tuning: every correct try is written with the problem pass rate)')
    p.add_argument('--gsm_temp', type=float, default=0.8)
    p.add_argument('--gsm_selfgen_budget', type=float, default=0, help='selfgen: stop after this many seconds (0 = no limit)')
    p.add_argument('--shard', default='0/1', help='k/n: this process handles every n-th problem starting at k')
    p.add_argument('--gen_bs', type=int, default=16)
    p.add_argument('--max_new', type=int, default=512)
    p.add_argument('--train_file'); p.add_argument('--val_file')
    p.add_argument('--train_n', type=int, default=1000, help='training examples (sst2)')
    p.add_argument('--eval_n', type=int, default=1000, help='validation examples (rte: 277 = all, sst2: 872 = all)')
    p.add_argument('--epochs', type=float, default=2)
    p.add_argument('--bs', type=int, default=8)
    p.add_argument('--max_len', type=int, default=192)
    p.add_argument('--lr', type=float, default=None, help='lora / qat: AdamW lr (2e-4 / 3e-5); latent_sgd: step per unit vote, in ternary steps (0.01)')
    p.add_argument('--clip', type=float, default=1.0, help='per-layer gradient norm clip (latent_sgd)')
    p.add_argument('--lora_r', type=int, default=16)
    p.add_argument('--lora_alpha', type=float, default=32)
    p.add_argument('--tau', type=float, default=32)
    p.add_argument('--tau_mode', default='fixed', choices=['fixed', 'adaptive'],
                   help='counter, adaptive: every --tau_win steps compare the mean training loss of the last window with '
                        'the one before (z-score from their own noise): clearly lower -> tau x0.95, no clear change -> '
                        'unchanged, clearly higher -> tau x1.10; tau stays in [--tau_min (default: the start value), --tau_max]')
    p.add_argument('--tau_win', type=int, default=20)
    p.add_argument('--qat_scale', default='fixed', choices=['fixed', 'absmean', 'learned'],
                   help='qat: fixed = checkpoint scale; absmean = recomputed from the latent weights every step '
                        '(the BitNet b1.58 recipe); learned = one trainable scale per matrix (LSQ-style)')
    p.add_argument('--qat_dtype', default='fp32', choices=['fp32', 'bf16'], help='qat: dtype of the latent weights (and their gradient)')
    p.add_argument('--qat_opt', default='adamw', choices=['adamw', 'adamw8bit'], help='qat: AdamW or 8-bit AdamW (bitsandbytes)')
    p.add_argument('--qat_init', default='grid', choices=['grid', 'master'],
                   help='qat: start the latent weights on the ternary grid, or at the full-precision master weights '
                        'when the checkpoint has them (BitNet bf16); TriLM has none, so it always starts on the grid')
    p.add_argument('--tau_min', type=float, default=0, help='0 = the starting tau')
    p.add_argument('--tau_max', type=float, default=127)
    p.add_argument('--leak', type=float, default=0.0, help='counter: per-step stochastic decay of each counter toward 0 '
                   '(fading memory as in Bop, still int8); window ~ 1/leak steps')
    p.add_argument('--vote_scale', type=float, default=0.1, help='counter learning rate: votes are scaled before stochastic rounding')
    p.add_argument('--grad_ternary', type=int, default=0, help='round the output gradient to {-1,0,+1} x one scale per token (unbiased, stochastic) '
                   'before the weight gradient: simulates multiplication-free training')
    p.add_argument('--sr_lr', type=float, default=0.0, help='sr arm: expected notches moved per unit normalized vote; '
                   '0 = vote_scale / tau, the same expected drift as the counter (memoryless stochastic rounding, as in DQT)')
    p.add_argument('--load_dtype', default='bf16', choices=['bf16', 'fp16', 'fp32'], help='dtype the checkpoint is read in')
    p.add_argument('--act_quant', type=int, default=1, help='int8 per-token activations (BitNet b1.58); 0 for models trained '
                   'without it (TriLM)')
    p.add_argument('--reset', default='zero', choices=['zero', 'carry'], help='counter after a flip: back to 0, or keep the '
                   'remainder (c -= tau): ablation of the reset')
    p.add_argument('--tune_n', type=int, default=0, help='classification: hold out this many training examples (never trained on)')
    p.add_argument('--tune', type=int, default=0, help='1 = evaluate on the held-out training examples instead of the validation '
                   'set (hyper-parameter selection without touching the test data)')
    p.add_argument('--fp16_matmul', type=int, default=0, help='GPU: run the ternary matmuls on fp16 tensor cores (ternary weights '
                   'are exact in fp16; gradients are rescaled first so they do not underflow)')
    p.add_argument('--kappa', type=float, default=0.5, help='counter init from the bf16 master weights (0 = zero)')
    p.add_argument('--topk_frac', type=float, default=3.7e-5, help='topk control: accumulate the normalized votes over the whole '
                   'run without changing any weight, then move the top fraction of weights (global, by |sum|) one step')
    p.add_argument('--bop_gamma', type=float, default=0.01, help='bop: moving-average rate of the accumulator')
    p.add_argument('--bop_thr', type=float, default=3.0, help='bop: threshold on the moving average (in normalized-vote units)')
    p.add_argument('--threads', type=int, default=0, help='0 = number of physical cores')
    p.add_argument('--eval_every', type=int, default=0)
    p.add_argument('--also_eval', default='', help='comma-separated tasks to score after training without training on them '
                   '(forgetting check), e.g. sst2,agnews')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--lr_sched', default='const', choices=['const', 'linear'], help='AdamW arms: constant lr, or 3%% '
                   'linear warmup then linear decay to 0 (the Low-Rank Ternary Adaptation recipe)')
    p.add_argument('--device', default='auto', help='auto | cpu | cuda')
    p.add_argument('--cache', type=int, default=-1, help='keep fp32 copies of the ternary weights (1 = faster on CPU, needs ~8 GB more); -1 = auto: on for CPU')
    p.add_argument('--max_steps', type=int, default=0, help='stop after this many steps (timing runs)')
    p.add_argument('--grad_ckpt', type=int, default=1, help='recompute each layer in backward instead of storing its activations (much less memory, ~30%% slower)')
    p.add_argument('--skip_eval', action='store_true', help='no evaluation (timing runs)')
    return p.parse_args()


def rss_gb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2**30
    except Exception:
        return float('nan')


# ----------------------------------------------------------------------------- ternary layers
def act_quant(x):
    """Per-token absmax int8 (BitNet b1.58 activations); identity gradient (STE)."""
    s = 127.0 / x.detach().abs().amax(-1, keepdim=True).clamp(min=1e-5)
    q = (x.detach() * s).round().clamp(-128, 127) / s
    return x + (q - x.detach())


def ternarize(W):
    beta = W.abs().mean().clamp(min=1e-5)
    return (W / beta).round().clamp(-1, 1).to(torch.int8), beta


def tlinear(xq, layer):
    """xq @ W_eff^T, on fp16 tensor cores when --fp16_matmul (fp32 accumulation, fp32 result)."""
    if layer.args.fp16_matmul and xq.is_cuda and layer.mode != 'qat':
        return F.linear(xq.half(), layer.weight_half()).float()
    return F.linear(xq, layer.weight_eff())


def ternary_split(W):
    """Ternary weights + scale(s). A checkpoint that is already ternary-valued (TriLM: t * s with one scale per row
    or per column block) is split exactly; otherwise (BitNet bf16 master weights) absmean ternarization."""
    A = W.abs()
    nz = A > 0
    if not nz.any():
        return ternarize(W)
    if torch.unique(A[nz][:200000]).numel() > 16:          # quick test on a sample: master weights, not ternary
        return ternarize(W)
    vals = torch.unique(A[nz])
    if len(vals) > 16:
        return ternarize(W)
    t = torch.sign(W).to(torch.int8)
    if len(vals) == 1:
        return t, vals[0]
    big = torch.where(nz, A, torch.full_like(A, float('inf')))
    for dim in (1, 0):                                   # one scale per row, else per column
        hi, lo = A.amax(dim), big.amin(dim)
        has = nz.any(dim)
        if torch.equal(hi[has], lo[has]):
            beta = torch.where(has, hi, hi[has].median())
            return t, (beta[:, None] if dim == 1 else beta[None, :]).contiguous()
    return ternarize(W)


class FusedTernary(torch.autograd.Function):
    """y = xq @ W_eff^T. In backward the layer's weight gradient is used immediately to update the
    layer and then discarded, so the full model gradient never exists in memory."""
    @staticmethod
    def forward(ctx, xq, anchor, layer):
        ctx.layer = layer
        ctx.save_for_backward(xq)
        return tlinear(xq, layer)

    @staticmethod
    def backward(ctx, go):
        (xq,) = ctx.saved_tensors
        layer = ctx.layer
        fp16 = layer.args.fp16_matmul and go.is_cuda
        if fp16:                                          # per-token rescale: gradients are tiny and would underflow in fp16
            a = go.abs().amax(-1, keepdim=True).clamp(min=1e-30)
            gx = ((go / a).half() @ layer.weight_half()).float() * a
        else:
            gx = go @ layer.weight_eff()                  # uses the weights as they were in forward
        g = go.reshape(-1, go.shape[-1])
        if layer.args.grad_ternary:
            scale = g.abs().amax(-1, keepdim=True).clamp(min=1e-30)     # per token
            p = g.abs() / scale                                         # in [0, 1]
            g = torch.sign(g) * (torch.rand_like(p) < p).to(g.dtype) * scale   # E[g] unchanged
        x2 = xq.reshape(-1, xq.shape[-1])
        if fp16:                                          # one scale per tensor (it must factor out of the token sum)
            G, X = g.abs().amax().clamp(min=1e-30), x2.abs().amax().clamp(min=1e-30)
            gW = ((g / G).half().t() @ (x2 / X).half()).float() * (G * X)
        else:
            gW = g.t() @ x2
        layer.update(gW)
        return gx, torch.zeros(1, dtype=go.dtype, device=go.device), None


def tern_qdq(X):
    """Per-tensor absmean ternarization, quantize-dequantize: values {-m, 0, +m} with m = mean|X|."""
    m = X.abs().mean().clamp(min=1e-5)
    return (X / m).round().clamp(-1, 1) * m, m


def closest_factors(x):
    a = int(math.sqrt(x))
    while x % a:
        a -= 1
    return a, x // a


class TAdaptFn(torch.autograd.Function):
    """Low-Rank Ternary Adaptation (Manolache et al. 2026), re-implemented: y = xq @ (M * W)^T with
    W = t * beta frozen and M = tern(A) kron tern(B), A (p x q) and B (r x s) fp32 latent factors trained through a
    straight-through estimator. The full weight gradient is formed one layer at a time in backward and reduced to
    the two small factor gradients, so it is never stored."""
    @staticmethod
    def forward(ctx, xq, Ka, Kb, layer):
        Aq, _ = tern_qdq(Ka); Bq, _ = tern_qdq(Kb)
        ctx.layer = layer
        ctx.save_for_backward(xq, Aq, Bq)
        W = torch.kron(Aq, Bq) * layer.weight_eff()
        if layer.args.fp16_matmul and xq.is_cuda:
            return F.linear(xq.half(), W.half()).float()
        return F.linear(xq, W)

    @staticmethod
    def backward(ctx, go):
        xq, Aq, Bq = ctx.saved_tensors
        layer = ctx.layer
        Wb = layer.weight_eff()
        W = torch.kron(Aq, Bq) * Wb
        g = go.reshape(-1, go.shape[-1]); x2 = xq.reshape(-1, xq.shape[-1])
        if layer.args.fp16_matmul and go.is_cuda:         # same rescaling as the counter's fused backward
            a = go.abs().amax(-1, keepdim=True).clamp(min=1e-30)
            gx = ((go / a).half() @ W.half()).float() * a
            G, X = g.abs().amax().clamp(min=1e-30), x2.abs().amax().clamp(min=1e-30)
            gW = ((g / G).half().t() @ (x2 / X).half()).float() * (G * X)
        else:
            gx = go @ W
            gW = g.t() @ x2
        gM = (gW * Wb).view(layer.kp, layer.kr, layer.kq, layer.ks)   # M[i*r+k, j*s+l] = A[i,j] * B[k,l]
        gA = torch.einsum('ikjl,kl->ij', gM, Bq)
        gB = torch.einsum('ikjl,ij->kl', gM, Aq)
        return gx, gA, gB, None                          # STE: d tern(A) / dA = 1


class FrozenTernary(torch.autograd.Function):
    """y = xq @ W_eff^T with frozen ternary weights: the weights are rebuilt in backward instead of
    being stored for it (storing them costs a full fp32 copy of the model)."""
    @staticmethod
    def forward(ctx, xq, layer):
        ctx.layer = layer
        return F.linear(xq, layer.weight_eff())

    @staticmethod
    def backward(ctx, go):
        return go @ ctx.layer.weight_eff(), None


class TernaryLinear(nn.Module):
    def __init__(self, lin, mode, args):
        super().__init__()
        W = lin.weight.data.float()
        self.mode, self.args = mode, args
        self.dout, self.din = W.shape
        self.flips = 0
        if mode == 'latent_sgd':
            self.register_buffer('W', W, persistent=False)  # fp32 master copy, updated in backward
        else:
            t, beta = ternary_split(W)
            self.split_err = float((t.float() * beta - W).abs().max() / W.abs().max().clamp(min=1e-30))
            self.register_buffer('t', t)
            if beta.numel() > 1:
                self.register_buffer('beta', beta.float())   # per-row / per-column scales (moves with the model)
            else:
                self.beta = float(beta)
            if mode == 'qat':
                # standard QAT baseline: fp32 latent weights + AdamW, straight-through rounding, scales kept fixed
                # (as the counter keeps them); initialised exactly at the checkpoint's ternary weights
                master = args.qat_init == 'master' and self.split_err > 1e-3     # the checkpoint holds real masters
                if master:
                    W0 = W.clone()
                elif args.qat_scale == 'absmean':
                    # grid start: scale the latents up by 1/(share of nonzero weights) so that mean|Wl| equals the
                    # checkpoint scale; otherwise absmean shrinks every layer by that share before any training
                    W0 = t.float() * beta / max(float((t != 0).float().mean()), 1e-3)
                else:
                    W0 = t.float() * beta
                self.Wl = nn.Parameter(W0)
                if args.qat_scale == 'learned':
                    self.Qs = nn.Parameter(torch.tensor(float(beta.mean()) if torch.is_tensor(beta) else float(beta)))
                self.t_init = t.clone()
            if mode == 'bop':
                self.register_buffer('m', torch.zeros_like(W, dtype=torch.float16))   # 16-bit moving average
                self.flipped = []
            if mode == 'sr':
                self.flipped = []
                self.t_init = t.clone()                      # CPU copy: net change measured at the end
            if mode == 'topk':
                self.register_buffer('acc', torch.zeros_like(W))   # fp32 sum of normalized votes (control arm only)
                self.flipped = []
                self.t_init = t.clone()
            if mode == 'counter':
                r = (W / beta - t.float()).clamp(-0.5, 0.5)  # 0 for an already-ternary checkpoint (TriLM)  # how close each weight was to flipping
                lim = max(1, args.tau - 1)
                c = (args.kappa * 2 * r * args.tau).round().clamp(-lim, lim).clamp(-127, 127)
                self.register_buffer('c', c.to(torch.int8))
                self.flipped = []
        if mode in ('counter', 'bop', 'sr', 'topk', 'latent_sgd'):
            self.anchor = nn.Parameter(torch.zeros(1))      # lets gradients flow through frozen embeddings
        if mode == 'tadapt':
            # Kronecker factor shapes as balanced as possible (dout = p*r, din = q*s); 'normalized' init as in the
            # authors' BitNet / GSM8K setup: |entries| ~ U(0.6, 1.4) with random signs, rows scaled to mean |.| = 1,
            # and the base ternary weights sign-compensated so that the adapted layer starts exactly at the checkpoint
            (self.kp, self.kr), (self.kq, self.ks) = closest_factors(self.dout), closest_factors(self.din)
            def init(shape):
                x = torch.empty(shape).uniform_(0.6, 1.4) * (torch.randint(0, 2, shape) * 2 - 1)
                return x / x.abs().mean(-1, keepdim=True)
            self.Ka, self.Kb = nn.Parameter(init((self.kp, self.kq))), nn.Parameter(init((self.kr, self.ks)))
            self.t_init = t.clone()                          # the checkpoint's weights, to measure the net change
            sA = torch.sign(self.Ka.data).to(torch.int8).view(self.kp, 1, self.kq, 1)
            sB = torch.sign(self.Kb.data).to(torch.int8).view(1, self.kr, 1, self.ks)
            self.t.copy_((self.t.view(self.kp, self.kr, self.kq, self.ks) * sA * sB).view(self.dout, self.din))
        if mode == 'lora':
            r = args.lora_r
            self.A = nn.Parameter(torch.randn(r, self.din) / math.sqrt(self.din))
            self.B = nn.Parameter(torch.zeros(self.dout, r))
            self.scale = args.lora_alpha / r

    def beta_flat(self, idx):
        if torch.is_tensor(self.beta):
            return self.beta.expand(self.dout, self.din).reshape(-1)[idx]
        return self.beta

    def weight_eff(self):
        if self.mode == 'qat':
            qs = self.args.qat_scale
            Wl = self.Wl.float()                                  # bf16 latents: math in fp32, gradient cast back
            if qs == 'learned':                                   # LSQ-style: gradients reach both Wl and the scale
                b = self.Qs.abs().clamp(min=1e-6)
                q = Wl / b
                return (q + (q.round().clamp(-1, 1) - q).detach()) * b
            b = Wl.detach().abs().mean().clamp(min=1e-5) if qs == 'absmean' else self.beta
            Wq = (Wl / b).round().clamp(-1, 1) * b
            return Wl + (Wq - Wl).detach()                     # straight-through estimator
        if self.mode == 'latent_sgd':
            t, beta = ternarize(self.W)
            return t.float() * beta
        if getattr(self, 'Wc', None) is not None:
            return self.Wc
        return self.t.float() * self.beta

    @torch.no_grad()
    def weight_half(self):
        """fp16 weights (t * beta). Cached while the layer is in eval mode (generation calls each layer once per
        token); never cached in training, where the weights change."""
        if self.mode in ('latent_sgd', 'qat'):
            return self.weight_eff().half()
        Wh = getattr(self, '_Wh', None)
        if Wh is None:
            Wh = (self.t.half() * self.beta).half()
            if not self.training:
                self._Wh = Wh
        return Wh

    def train(self, mode=True):
        self._Wh = None
        return super().train(mode)

    @torch.no_grad()
    def build_cache(self):
        self.Wc = self.t.float() * self.beta

    def forward(self, x):
        xq = act_quant(x) if self.args.act_quant else x
        if self.mode in ('counter', 'bop', 'sr', 'topk', 'latent_sgd') and self.training and torch.is_grad_enabled():
            return FusedTernary.apply(xq, self.anchor, self)
        if self.mode == 'tadapt':
            return TAdaptFn.apply(xq, self.Ka, self.Kb, self)
        if self.mode == 'lora' and torch.is_grad_enabled() and xq.requires_grad:
            y = FrozenTernary.apply(xq, self)
        else:
            y = tlinear(xq, self)
        if self.mode == 'lora':
            y = y + (x @ self.A.t()) @ self.B.t() * self.scale
        return y

    @torch.no_grad()
    def update(self, gW):
        a = self.args
        self._Wh = None
        if self.mode == 'latent_sgd':
            # same normalized step as the counter, accumulated in fp32 instead of an int8 counter:
            # each weight moves lr * beta per unit vote (lr = fraction of one ternary step)
            t_old, beta = ternarize(self.W)
            s = gW / gW.pow(2).mean().sqrt().clamp(min=1e-30)
            self.W.add_(s, alpha=-a.lr * float(beta))
            self.flips += int((ternarize(self.W)[0] != t_old).sum())
            return
        if self.mode == 'bop':
            # ternary Bop (Helwegen et al. 2019): moving average of the normalized votes, one notch per step
            # while it stays past the threshold, no reset
            s = -gW / gW.pow(2).mean().sqrt().clamp(min=1e-30)
            m = self.m.float().mul_(1 - a.bop_gamma).add_(s, alpha=a.bop_gamma)
            t = self.t.float()
            nt = torch.where(m >= a.bop_thr, (t + 1).clamp(max=1), torch.where(m <= -a.bop_thr, (t - 1).clamp(min=-1), t))
            changed = nt != t
            k = int(changed.sum())
            if k:
                idx = changed.flatten().nonzero().squeeze(1)
                self.flipped.append(idx.to(torch.int32).cpu()); self.flips += k
                if getattr(self, 'Wc', None) is not None:
                    self.Wc.view(-1)[idx] = nt.flatten()[idx] * self.beta_flat(idx)
            self.m.copy_(m.to(torch.float16)); self.t.copy_(nt.to(torch.int8))
            return
        if self.mode == 'topk':
            # control: same normalized vote as the counter, summed in fp32, no weight changes until the end
            self.acc.add_(-gW / gW.pow(2).mean().sqrt().clamp(min=1e-30))
            return
        if self.mode == 'sr':
            # stochastic rounding straight onto the ternary grid (DQT-style, no memory): move one notch with
            # probability |d|; same normalized vote as the counter, same expected drift by default
            d = -(a.sr_lr or a.vote_scale / a.tau) * gW / gW.pow(2).mean().sqrt().clamp(min=1e-30)
            move = torch.rand_like(d) < d.abs().clamp(max=1)
            t = self.t.float()
            nt = torch.where(move & (d > 0), (t + 1).clamp(max=1), torch.where(move & (d < 0), (t - 1).clamp(min=-1), t))
            changed = nt != t
            k = int(changed.sum())
            if k:
                idx = changed.flatten().nonzero().squeeze(1)
                self.flips += k                         # many moves: positions not logged, net change from t_init
                if getattr(self, 'Wc', None) is not None:
                    self.Wc.view(-1)[idx] = nt.flatten()[idx] * self.beta_flat(idx)
                self.t.copy_(nt.to(torch.int8))
            return
        s = -a.vote_scale * gW / gW.pow(2).mean().sqrt().clamp(min=1e-30)
        cf = self.c.float()
        if a.leak:                                         # E[c] shrinks by (1 - leak) per step, stays an integer
            cf = cf - torch.sign(cf) * (torch.rand_like(cf) < (a.leak * cf.abs()).clamp(max=1)).float()
        cf = (cf + torch.floor(s + torch.rand_like(s))).clamp(-127, 127)
        t = self.t.float()
        up, dn = cf >= a.tau, cf <= -a.tau
        nt = torch.where(up, (t + 1).clamp(max=1), torch.where(dn, (t - 1).clamp(min=-1), t))
        if a.reset == 'carry':                             # ablation: keep the remainder instead of resetting
            cf = torch.where(up, cf - a.tau, torch.where(dn, cf + a.tau, cf))
        else:
            cf = torch.where(up | dn, torch.zeros_like(cf), cf)
        changed = nt != t
        k = int(changed.sum())
        if k:
            idx = changed.flatten().nonzero().squeeze(1)
            self.flipped.append(idx.to(torch.int32).cpu())
            self.flips += k
            if getattr(self, 'Wc', None) is not None:
                self.Wc.view(-1)[idx] = nt.flatten()[idx] * self.beta_flat(idx)
        self.c.copy_(cf.to(torch.int8))
        self.t.copy_(nt.to(torch.int8))

    @torch.no_grad()
    def merge_tadapt(self):
        """W' = t * (tern(A) kron tern(B)) * beta * mean|A| * mean|B|: still one ternary matrix with one scale."""
        Aq, mA = tern_qdq(self.Ka.detach()); Bq, mB = tern_qdq(self.Kb.detach())
        sA = (Aq / mA).round().to(torch.int8).view(self.kp, 1, self.kq, 1)
        sB = (Bq / mB).round().to(torch.int8).view(1, self.kr, 1, self.ks)
        self.t.copy_((self.t.view(self.kp, self.kr, self.kq, self.ks) * sA * sB).view(self.dout, self.din))
        f = float(mA * mB)
        if torch.is_tensor(self.beta):
            self.beta.mul_(f)
        else:
            self.beta = self.beta * f
        self.mode = 'frozen'; self._Wh = None
        del self.Ka, self.Kb
        if getattr(self, 'Wc', None) is not None:
            self.build_cache()

    @torch.no_grad()
    def merge_lora(self, rescale=False):
        """Fold the adapter into the ternary weights. rescale=False keeps the layer's scale beta and rounds
        t + delta/beta back to {-1, 0, 1} (a weight changes only where the adapter pushed it past a rounding
        boundary). rescale=True re-ternarizes from scratch, recomputing beta = mean|W| (changes every layer's scale)."""
        if not hasattr(self, 't0'):                        # keep the adapter low-rank (a full fp32 delta per layer = ~8 GB)
            self.t0, self.beta0 = self.t.cpu().clone(), self.beta
            self.Am, self.Bm = self.A.detach().clone(), self.B.detach().clone()
            self.mode = 'frozen'
            del self.A, self.B
        W = self.t0.to(self.t.device).float() * self.beta0 + self.scale * (self.Bm @ self.Am)
        if rescale:
            t, beta = ternarize(W)
        else:
            t, beta = (W / self.beta0).round().clamp(-1, 1).to(torch.int8), self.beta0
        self.t.copy_(t); self._Wh = None
        if torch.is_tensor(self.beta):
            if not torch.is_tensor(beta) or beta.numel() == 1:
                self.beta.fill_(float(beta))
        else:
            self.beta = float(beta)
        if getattr(self, 'Wc', None) is not None:
            self.build_cache()


def build(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'right'
    tok.truncation_side = 'left'                         # keep the question and 'Answer:' at the end
    try:
        dt = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}[args.load_dtype]
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dt, low_cpu_mem_usage=True)
    except TypeError:                                     # older transformers
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dt, low_cpu_mem_usage=True)
    model.config.use_cache = False
    mode = {'zeroshot': 'frozen'}.get(args.arm, args.arm)
    layers = []
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            w = getattr(child, 'weight', None)
            if cname in TARGETS and isinstance(w, torch.Tensor) and w.dim() == 2:
                tl = TernaryLinear(child, mode, args)
                setattr(mod, cname, tl)
                layers.append(tl)
                del child
    assert layers, 'no q/k/v/o/gate/up/down projections found: check the model architecture'
    # transformers marks BitNet as a 'quantized' model and blocks .float()/.to(); its online-quantized
    # linear layers have just been replaced by ours, so the plain torch conversion is safe here
    nn.Module.float(model)                                # fp32 math; int8 buffers stay int8
    if args.arm == 'qat' and args.qat_dtype == 'bf16':
        for l in layers:
            l.Wl.data = l.Wl.data.to(torch.bfloat16)
    model.requires_grad_(False)
    for l in layers:
        for n in ('anchor', 'A', 'B', 'Wl', 'Ka', 'Kb', 'Qs'):
            if hasattr(l, n):
                getattr(l, n).requires_grad_(True)
    return model, tok, layers


# ----------------------------------------------------------------------------------------- data
DEV = torch.device('cpu')


def glue(name):
    from datasets import load_dataset
    try:
        return load_dataset('nyu-mll/glue', name)
    except Exception:
        return load_dataset('glue', name)


def cls_data(args):
    """Returns (train, val) as lists of (prompt, label index)."""
    from datasets import load_dataset
    if args.task == 'sst2':
        ds = glue('sst2'); tr, va = ds['train'], ds['validation']
        fmt = lambda r: (f"Review: {r['sentence'].strip()}\nSentiment:", r['label'])
    elif args.task == 'rte':
        ds = glue('rte'); tr, va = ds['train'], ds['validation']       # label 0 = entailment, 1 = not
        fmt = lambda r: (f"{r['sentence1'].strip()}\nQuestion: Does this imply that \"{r['sentence2'].strip()}\"? Yes or No?\nAnswer:", r['label'])
    elif args.task == 'tunisian':
        # 49.9k Tunisian-dialect comments (Arabic script and Latin Arabizi), binary sentiment; one split, so a fixed
        # held-out test slice is cut here (deduplicated first, so no comment is in both)
        full = load_dataset('arbml/Tunisian_Dialect_Corpus', split='train')
        names = full.features['label'].names
        neg = [i for i, n in enumerate(names) if 'neg' in n.lower()][0]
        seen, keep = set(), []
        for i, t in enumerate(full['Tweet']):
            k = (t or '').strip().lower()
            if k and k not in seen:
                seen.add(k); keep.append(i)
        full = full.select(keep).shuffle(seed=1234)
        n_test = min(args.eval_n, 2000)
        va, tr = full.select(range(n_test)), full.select(range(n_test, len(full)))
        fmt = lambda r: (f"Comment (Tunisian dialect): {r['Tweet'].strip()}\nSentiment (negative or positive):",
                         0 if r['label'] == neg else 1)
    else:
        ds = load_dataset('fancyzhx/ag_news'); tr, va = ds['train'], ds['test']
        fmt = lambda r: (f"Article: {r['text'].strip()}\nTopic (World, Sports, Business or Science):", r['label'])
    if args.tune_n:                                       # fixed held-out slice, never trained on
        pool = tr.shuffle(seed=1234)
        held = pool.select(range(args.tune_n))
        tr = pool.select(range(args.tune_n, len(pool)))
        if args.tune:
            va = held
    tr = tr.shuffle(seed=args.seed).select(range(min(args.train_n, len(tr))))
    if args.tune:
        return [fmt(r) for r in tr], [fmt(r) for r in va]
    va = va.shuffle(seed=1234).select(range(min(args.eval_n, len(va)))) if args.task == 'agnews' else va.select(range(min(args.eval_n, len(va))))
    if args.task == 'tunisian':
        print(f'tunisian: {len(tr)} training comments available, {len(va)} test comments held out', flush=True)
    return [fmt(r) for r in tr], [fmt(r) for r in va]


def label_token(tok, word):
    """First token of the label word that carries text (sentencepiece may emit a bare '▁' first)."""
    for i in tok(word, add_special_tokens=False)['input_ids']:
        if tok.decode([i]).strip():
            return i
    raise ValueError(f'no token for label {word!r}')


def cls_logits(model, tok, batch, label_ids, args):
    enc = tok([p for p, _ in batch], return_tensors='pt', padding=True, truncation=True, max_length=args.max_len)
    ids, am = enc['input_ids'].to(DEV), enc['attention_mask'].to(DEV)
    h = model.base_model(input_ids=ids, attention_mask=am).last_hidden_state
    last = am.sum(1) - 1
    h = h[torch.arange(h.shape[0], device=DEV), last]    # hidden state at the last prompt token
    W = model.get_output_embeddings().weight[label_ids]  # only the label words: skips the 128k-vocab head
    return h @ W.t(), int(ids.numel())                     # tokens processed, padding included


def text_data(args, tok):
    def chunks(path):
        ids = tok(open(path, encoding='utf-8').read(), add_special_tokens=False)['input_ids']
        L = args.max_len
        return [ids[i:i + L] for i in range(0, len(ids) - L + 1, L)]
    return chunks(args.train_file), chunks(args.val_file)


def text_loss(model, batch):
    x = torch.tensor(batch, device=DEV)
    return model(input_ids=x, labels=x).loss, x.numel()


def example_len(ex):
    if isinstance(ex, dict):
        return len(ex['ids'])
    return len(ex[0]) if isinstance(ex, tuple) else len(ex)


def make_batches(train, bs, rng):
    """Shuffle, then group examples of similar length into the same batch (less padding),
    then shuffle the order of the batches."""
    order = list(range(len(train))); rng.shuffle(order)
    batches = []
    pool = bs * 16
    for i in range(0, len(order), pool):
        chunk = sorted(order[i:i + pool], key=lambda j: example_len(train[j]))
        batches += [chunk[k:k + bs] for k in range(0, len(chunk), bs)]
    rng.shuffle(batches)
    return batches


@torch.no_grad()
def evaluate(model, tok, val, label_ids, args):
    model.eval()
    order = sorted(range(len(val)), key=lambda j: example_len(val[j]))
    if args.task != 'text':
        K = len(label_ids); conf = np.zeros((K, K), dtype=np.int64)     # conf[true, predicted]
        for i in range(0, len(order), 16):
            b = [val[j] for j in order[i:i + 16]]
            lg, _ = cls_logits(model, tok, b, label_ids, args)
            for y, p in zip([y for _, y in b], lg.argmax(1).cpu().tolist()): conf[y, p] += 1
        f1 = [2 * conf[k, k] / max(conf[k].sum() + conf[:, k].sum(), 1) for k in range(K)]
        res = dict(accuracy=float(np.trace(conf) / conf.sum()), macro_f1=float(np.mean(f1)),
                   recall=[float(conf[k, k] / max(conf[k].sum(), 1)) for k in range(K)],
                   pred_frac=[float(conf[:, k].sum() / conf.sum()) for k in range(K)])
    else:
        tot = n = 0
        for i in range(0, len(val), args.bs):
            loss, k = text_loss(model, val[i:i + args.bs]); tot += loss.item() * k; n += k
        res = dict(val_loss=tot / n, ppl=math.exp(tot / n))
    model.train()
    return res


@torch.no_grad()
def wikitext_ppl(model, tok, args, n_chunks=64, L=512, bs=4):
    """Perplexity on the first n_chunks x L tokens of WikiText-2 test (fixed): a forgetting check that does not
    depend on the model being good at any classification task."""
    from datasets import load_dataset
    text = '\n\n'.join(t for t in load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='test')['text'] if t.strip())
    ids = tok(text[:400000], add_special_tokens=False)['input_ids']
    chunks = [ids[i:i + L] for i in range(0, len(ids) - L + 1, L)][:n_chunks]
    model.eval(); tot = n = 0.0
    for i in range(0, len(chunks), bs):
        loss, k = text_loss(model, chunks[i:i + bs])
        tot += loss.item() * k; n += k
    model.train()
    return math.exp(tot / n)


@torch.no_grad()
def topk_finalize(layers, frac):
    """Control arm: move the k = frac * (all ternary weights) weights with the largest |summed normalized vote| one
    step in the vote's direction (only where that step stays in {-1, 0, +1}); global threshold by bisection."""
    n_w = sum(l.dout * l.din for l in layers)
    k = max(1, int(round(frac * n_w)))
    def score(l):
        t, a = l.t.float(), l.acc
        ok = ((a > 0) & (t < 1)) | ((a < 0) & (t > -1))
        return torch.where(ok, a.abs(), torch.zeros_like(a))
    lo, hi = 0.0, max(float(score(l).max()) for l in layers)
    for _ in range(50):
        mid = (lo + hi) / 2
        if sum(int((score(l) > mid).sum()) for l in layers) > k: lo = mid
        else: hi = mid
    n = 0
    for l in layers:
        m = score(l) > hi
        n += int(m.sum())
        l.t.copy_(torch.where(m, (l.t.float() + torch.sign(l.acc)).clamp(-1, 1), l.t.float()).to(torch.int8))
        l.flips += int(m.sum()); l._Wh = None
        if getattr(l, 'Wc', None) is not None:
            l.build_cache()
    print(f'topk control: moved {n} weights ({n / n_w * 100:.4f}%, target {k})', flush=True)


MC_TASKS = ('arc_e', 'arc_c', 'hellaswag', 'obqa', 'piqa')


def mc_data(task, n=500):
    """(context, [choices], gold index) for zero-shot multiple-choice probes; fixed subsample of n."""
    from datasets import load_dataset
    if task in ('arc_e', 'arc_c'):
        ds = load_dataset('allenai/ai2_arc', 'ARC-Easy' if task == 'arc_e' else 'ARC-Challenge', split='test')
        rows = [(f"Question: {r['question']}\nAnswer:", [' ' + c for c in r['choices']['text']],
                 r['choices']['label'].index(r['answerKey'])) for r in ds if r['answerKey'] in r['choices']['label']]
    elif task == 'hellaswag':
        ds = load_dataset('Rowan/hellaswag', split='validation')
        clean = lambda x: x.replace(' [title]', '. ').replace('[header]', '').replace('  ', ' ').strip()
        rows = [(clean(r['activity_label'] + ': ' + r['ctx']), [' ' + clean(e) for e in r['endings']], int(r['label'])) for r in ds]
    elif task == 'obqa':
        ds = load_dataset('allenai/openbookqa', 'main', split='test')
        rows = [(r['question_stem'], [' ' + c for c in r['choices']['text']], r['choices']['label'].index(r['answerKey'])) for r in ds]
    else:
        ds = load_dataset('ybisk/piqa', split='validation', trust_remote_code=True)
        rows = [(f"Question: {r['goal']}\nAnswer:", [' ' + r['sol1'], ' ' + r['sol2']], int(r['label'])) for r in ds]
    rng = np.random.default_rng(1234)
    idx = rng.permutation(len(rows))[:n]
    return [rows[i] for i in idx]


@torch.no_grad()
def mc_eval(model, tok, task, n=500):
    """Zero-shot accuracy, choices scored by summed log-probability per character (acc_norm style)."""
    data = mc_data(task, n)
    model.eval(); correct = 0
    for ctx, choices, gold in data:
        c_ids = tok(ctx, add_special_tokens=True)['input_ids']
        seqs = [c_ids + tok(ch, add_special_tokens=False)['input_ids'] for ch in choices]
        L = max(len(x) for x in seqs)
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        inp = torch.tensor([x + [pad] * (L - len(x)) for x in seqs], device=DEV)
        att = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x in seqs], device=DEV)
        lp = torch.log_softmax(model(input_ids=inp, attention_mask=att).logits.float(), -1)
        scores = []
        for j, x in enumerate(seqs):
            tgt = torch.tensor(x[len(c_ids):], device=DEV)
            pos = torch.arange(len(c_ids) - 1, len(x) - 1, device=DEV)
            scores.append(float(lp[j, pos, tgt].sum()) / max(1, len(choices[j])))
        correct += int(int(np.argmax(scores)) == gold)
    model.train()
    return correct / len(data)


def eval_other_tasks(model, tok, args):
    """Score the model on other tasks it was not trained on (forgetting check)."""
    import copy
    out = {}
    for task in [t for t in args.also_eval.split(',') if t]:
        if task == 'wikitext':                           # general language modelling: perplexity, lower = better
            out[task] = wikitext_ppl(model, tok, args)
            print(f'ALSO_EVAL wikitext: perplexity {out[task]:.3f}', flush=True)
            continue
        if task in MC_TASKS:                             # zero-shot multiple choice (general knowledge / commonsense)
            try:
                out[task] = mc_eval(model, tok, task)
                print(f'ALSO_EVAL {task}: accuracy {out[task]:.4f}', flush=True)
            except Exception as e:
                print(f'ALSO_EVAL {task}: skipped ({type(e).__name__}: {str(e)[:150]})', flush=True)
            continue
        a2 = copy.copy(args); a2.task = task
        a2.eval_n = {'sst2': 872, 'rte': 277}.get(task, 500); a2.train_n = 1
        _, val = cls_data(a2)
        ids = [tok(w, add_special_tokens=False)['input_ids'][0] for w in TASKS[task]['labels']]
        out[task] = evaluate(model, tok, val, ids, a2)['accuracy']
        print(f'ALSO_EVAL {task}: accuracy {out[task]:.4f}', flush=True)
    return out


# ----------------------------------------------------------------------------------------- code
INSTR = 'Please provide a self-contained Python script that solves the following problem in a markdown code block:'
RESP = 'Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:\n```python\n'


def chat_prompt_text(tok, user):
    if getattr(tok, 'chat_template', None):
        return tok.apply_chat_template([{'role': 'user', 'content': user}], add_generation_prompt=True, tokenize=False)
    return f'User: {user}\nAssistant: '


def encode_chat(tok, user, assistant, max_len):
    p = tok(chat_prompt_text(tok, user), add_special_tokens=False)['input_ids']
    a = tok(assistant, add_special_tokens=False)['input_ids'] + [tok.eos_token_id]
    if len(p) + len(a) > max_len:
        return None                                       # drop instead of truncating the answer
    return dict(ids=p + a, n_prompt=len(p))


def benchmark_shingles():
    """10-word sequences and function names from every HumanEval+ and MBPP+ problem and reference solution."""
    from evalplus.data import get_human_eval_plus, get_mbpp_plus
    sh, names = set(), set()
    for ds in (get_human_eval_plus(), get_mbpp_plus()):
        for p in ds.values():
            w = re.findall(r'\w+', (p['prompt'] + '\n' + p.get('canonical_solution', '')).lower())
            sh.update(' '.join(w[i:i + 10]) for i in range(len(w) - 9))
            names.add(p['entry_point'])
    return sh, names


def code_data(args, tok):
    """Magicoder OSS-Instruct Python examples, minus anything overlapping the benchmarks."""
    from datasets import load_dataset
    ds = load_dataset('ise-uiuc/Magicoder-OSS-Instruct-75K', split='train').shuffle(seed=args.seed)
    sh, names = benchmark_shingles()
    keep, removed, too_long, scanned = [], 0, 0, 0
    need = args.train_n + args.code_dev
    for r in ds:
        sol = r['solution']
        if 'python' not in (r.get('lang') or '').lower() and '```python' not in sol:
            continue
        scanned += 1
        text = r['problem'] + '\n' + sol
        w = re.findall(r'\w+', text.lower())
        if any(' '.join(w[i:i + 10]) in sh for i in range(len(w) - 9)) or any(f'def {n}(' in text for n in names):
            removed += 1
            continue
        ex = encode_chat(tok, r['problem'], sol, args.max_len)
        if ex is None:
            too_long += 1
            continue
        keep.append(ex)
        if len(keep) >= need:
            break
    info = dict(python_scanned=scanned, removed_benchmark_overlap=removed, dropped_too_long=too_long, kept=len(keep))
    print(f'code data: {info}', flush=True)
    return keep[:args.train_n], keep[args.train_n:need], info


def _head_ce(hc, yc, W):
    return F.cross_entropy(hc @ W.t(), yc, reduction='sum')


def code_loss(model, batch, tok):
    """Next-token loss on the answer tokens only; the 128k-vocab head runs in recomputed chunks."""
    from torch.utils.checkpoint import checkpoint
    L = max(len(e['ids']) for e in batch)
    ids = torch.full((len(batch), L), tok.pad_token_id, dtype=torch.long)
    am = torch.zeros_like(ids); lab = torch.full_like(ids, -100)
    for i, e in enumerate(batch):
        n = len(e['ids'])
        ids[i, :n] = torch.tensor(e['ids']); am[i, :n] = 1
        lab[i, e['n_prompt']:n] = ids[i, e['n_prompt']:n]
    ids, am, lab = ids.to(DEV), am.to(DEV), lab.to(DEV)
    h = model.base_model(input_ids=ids, attention_mask=am).last_hidden_state[:, :-1]
    y = lab[:, 1:]
    sel = y != -100
    hs, ys = h[sel], y[sel]
    W = model.get_output_embeddings().weight
    tot = 0.0
    for i in range(0, len(ys), 512):
        if torch.is_grad_enabled():
            tot = tot + checkpoint(_head_ce, hs[i:i + 512], ys[i:i + 512], W, use_reentrant=False)
        else:
            tot = tot + _head_ce(hs[i:i + 512], ys[i:i + 512], W)
    return tot / max(len(ys), 1), int(ids.numel())


@torch.no_grad()
def code_dev_loss(model, tok, dev, args):
    model.eval(); tot = n = 0.0
    for i in range(0, len(dev), args.bs):
        b = dev[i:i + args.bs]
        loss, _ = code_loss(model, b, tok)
        k = sum(len(e['ids']) - e['n_prompt'] for e in b)
        tot += loss.item() * k; n += k
    model.train()
    return tot / max(n, 1)


@torch.no_grad()
def code_eval(model, tok, args, tag):
    """Greedy generation for every HumanEval+ / MBPP+ problem, scored by EvalPlus (runs the code in subprocesses)."""
    import subprocess, sys
    from evalplus.data import get_human_eval_plus, get_mbpp_plus, write_jsonl
    model.eval(); tok.padding_side = 'left'; res = {}
    for bench in [b for b in args.code_eval.split(',') if b]:
        probs = get_human_eval_plus() if bench == 'humaneval' else get_mbpp_plus()
        items = list(probs.items())
        if args.code_eval_n:
            items = items[:args.code_eval_n]
        prompts = [chat_prompt_text(tok, f"{INSTR}\n```python\n{p['prompt'].strip()}\n```") + RESP for _, p in items]
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        sols = [None] * len(items); t0 = time.time()
        for k in range(0, len(order), args.gen_bs):
            idx = order[k:k + args.gen_bs]
            enc = tok([prompts[i] for i in idx], return_tensors='pt', padding=True, add_special_tokens=False).to(DEV)
            out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False, use_cache=True,
                                 pad_token_id=tok.pad_token_id, stop_strings=['```'], tokenizer=tok)
            for j, i in enumerate(idx):
                text = tok.decode(out[j, enc['input_ids'].shape[1]:], skip_special_tokens=True)
                sols[i] = text.split('```')[0]
        path = os.path.join(args.out, f'{tag}_{bench}_samples.jsonl')
        write_jsonl(path, [dict(task_id=tid, solution=sol) for (tid, _), sol in zip(items, sols)])
        r = dict(generation_seconds=time.time() - t0, samples=path)
        if not args.code_eval_n:
            ev = subprocess.run([sys.executable, '-m', 'evalplus.evaluate', '--dataset', bench, '--samples', path],
                                capture_output=True, text=True)
            passes = re.findall(r'pass@1:\s*([0-9.]+)', ev.stdout)
            r.update(base=float(passes[0]) if passes else None, plus=float(passes[1]) if len(passes) > 1 else None)
            if not passes:
                r['evalplus_output'] = (ev.stdout + ev.stderr)[-1500:]
        else:
            r['example'] = sols[0][:300]
        res[bench] = r
        print(f'CODE_EVAL [{tag}] {bench}: ' + json.dumps({k: v for k, v in r.items() if k != 'samples'}), flush=True)
    tok.padding_side = 'right'; model.train()
    return res


GEN_TASKS = ('code', 'gsm8k')
GSM_INSTR = 'Solve this math word problem step by step. End with a line of the form "#### <final numeric answer>".'


def gsm_clean(answer):
    return re.sub(r'<<[^>]*>>', '', answer).strip()     # drop the calculator annotations


def gsm_number(text):
    m = re.findall(r'####\s*\$?\s*(-?[\d,]*\.?\d+)', text)
    if not m:
        m = re.findall(r'-?[\d,]*\.?\d+', text)
    if not m:
        return None
    try:
        return float(m[-1].replace(',', ''))
    except ValueError:
        return None


def gsm_data(args, tok):
    """GSM8K official training split (the test split is only used for scoring); answer-only loss."""
    from datasets import load_dataset
    if args.gsm_train_file:
        rows = [json.loads(l) for l in open(args.gsm_train_file)]
        random.Random(args.seed).shuffle(rows)
        ds = [dict(question=r['question'], answer=r['answer']) for r in rows]
    else:
        ds = load_dataset('openai/gsm8k', 'main', split='train').shuffle(seed=args.seed)
    keep, too_long = [], 0
    need = args.train_n + args.code_dev
    for r in ds:
        ans = r['answer'] if args.gsm_train_file else gsm_clean(r['answer'])
        ex = encode_chat(tok, f"{GSM_INSTR}\n\n{r['question']}", ans, args.max_len)
        if ex is None:
            too_long += 1
            continue
        keep.append(ex)
        if len(keep) >= need:
            break
    info = dict(source=args.gsm_train_file or 'openai/gsm8k train', dropped_too_long=too_long, kept=len(keep))
    print(f'gsm8k data: {info}', flush=True)
    return keep[:args.train_n], keep[args.train_n:need], info


def gsm_eval(model, tok, args, tag):
    """Greedy chain-of-thought answers for the first gsm_n GSM8K test problems; exact numeric match."""
    from datasets import load_dataset
    test = load_dataset('openai/gsm8k', 'main', split='test')
    n = args.code_eval_n or args.gsm_n
    items = [test[i] for i in range(min(n, len(test)))]
    model.eval(); tok.padding_side = 'left'
    prompts = [chat_prompt_text(tok, f"{GSM_INSTR}\n\n{r['question']}") for r in items]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    outs = [None] * len(items); t0 = time.time()
    for k in range(0, len(order), args.gen_bs):
        idx = order[k:k + args.gen_bs]
        enc = tok([prompts[i] for i in idx], return_tensors='pt', padding=True, add_special_tokens=False).to(DEV)
        out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False, use_cache=True, pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idx):
            outs[i] = tok.decode(out[j, enc['input_ids'].shape[1]:], skip_special_tokens=True)
    gold = [gsm_number(r['answer']) for r in items]
    pred = [gsm_number(o) for o in outs]
    correct = sum(p is not None and g is not None and abs(p - g) < 1e-6 for p, g in zip(pred, gold))
    path = os.path.join(args.out, f'{tag}_gsm8k_generations.jsonl')
    with open(path, 'w') as f:
        for r, o, p_ in zip(items, outs, pred):
            f.write(json.dumps(dict(question=r['question'], gold=gsm_number(r['answer']), pred=p_, output=o)) + '\n')
    res = dict(gsm8k=dict(accuracy=correct / len(items), n=len(items), no_answer=sum(p is None for p in pred),
                          generation_seconds=time.time() - t0, example=outs[0][:300]))
    print(f'GSM8K [{tag}]: ' + json.dumps(res['gsm8k']), flush=True)
    tok.padding_side = 'right'; model.train()
    return res


def gsm_selfgen(model, tok, args):
    """Self-distillation data: the model answers GSM8K *training* problems in its own style; only answers whose final
    number matches the gold answer are kept. The test split is never touched."""
    from datasets import load_dataset
    train = load_dataset('openai/gsm8k', 'main', split='train').shuffle(seed=4321)
    k, n = (int(x) for x in args.shard.split('/'))
    idx = [i for i in range(min(args.gsm_selfgen, len(train))) if i % n == k]
    items = [train[i] for i in idx]
    model.eval(); tok.padding_side = 'left'
    prompts = [chat_prompt_text(tok, f"{GSM_INSTR}\n\n{r['question']}") for r in items]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    S = max(1, args.gsm_samples); per = max(1, args.gen_bs // S)       # prompts per batch (S tries each)
    torch.manual_seed(args.seed * 1000 + k)
    outs = [[] for _ in items]; t0 = time.time()
    starts = list(range(0, len(order), per))
    random.Random(k).shuffle(starts)               # batches of similar length, in random order (a time stop is unbiased)
    for b, j0 in enumerate(starts):
        if args.gsm_selfgen_budget and time.time() - t0 > args.gsm_selfgen_budget:
            print(f'selfgen: time budget reached after {b}/{len(starts)} batches', flush=True); break
        sel = order[j0:j0 + per]
        enc = tok([prompts[i] for i in sel], return_tensors='pt', padding=True, add_special_tokens=False).to(DEV)
        kw = dict(do_sample=True, temperature=args.gsm_temp, top_p=0.95, num_return_sequences=S) if S > 1 else dict(do_sample=False)
        out = model.generate(**enc, max_new_tokens=args.max_new, use_cache=True, pad_token_id=tok.pad_token_id, **kw)
        L = enc['input_ids'].shape[1]
        for j, i in enumerate(sel):
            outs[i] = [tok.decode(out[j * S + s, L:], skip_special_tokens=True).strip() for s in range(S)]
        if b % 10 == 0:
            print(f'selfgen batch {b + 1}/{len(starts)} | {time.time() - t0:.0f}s', flush=True)
    done = sum(bool(o) for o in outs)
    good = solved = tries_ok = 0
    with open(args.gsm_selfgen_out, 'w') as f:
        for i, (r, os_) in enumerate(zip(items, outs)):
            g = gsm_number(r['answer'])
            ok = [o for o in os_ if '####' in o and g is not None and gsm_number(o) is not None and abs(g - gsm_number(o)) < 1e-6]
            tries_ok += len(ok); solved += bool(ok)
            for o in dict.fromkeys(ok):                      # distinct correct answers only
                f.write(json.dumps(dict(question=r['question'], answer=o, pid=idx[i], pass_rate=len(ok) / S)) + '\n'); good += 1
    res = dict(selfgen=dict(problems=done, tries=S, kept_correct=good, solved_problems=solved,
                            correct_try_frac=tries_ok / max(1, done * S), seconds=time.time() - t0))
    print('RESULT', json.dumps(res), flush=True)
    return res


def gen_data(args, tok):
    return code_data(args, tok) if args.task == 'code' else gsm_data(args, tok)


def gen_eval(model, tok, args, tag):
    return code_eval(model, tok, args, tag) if args.task == 'code' else gsm_eval(model, tok, args, tag)


def mem_gb():
    m = rss_gb()
    if DEV.type == 'cuda':
        return max(m if m == m else 0, torch.cuda.max_memory_allocated() / 2**30)
    return m


# ----------------------------------------------------------------------------------------- main
def main():
    global DEV
    args = get_args()
    os.makedirs(args.out, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.lr is None:
        args.lr = {'lora': 2e-4, 'qat': 3e-5, 'latent_sgd': 0.01}.get(args.arm, 0.0)
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    DEV = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False         # true fp32 on GPU: same math as the CPU runs
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False   # fp16 matmuls accumulate in fp32
    threads = args.threads or max(1, (os.cpu_count() or 2) // 2)
    torch.set_num_threads(threads)
    if args.cache == -1:
        args.cache = int(DEV.type == 'cpu')
    print(f'arm {args.arm} | device {DEV} | {threads} CPU threads | weight cache {bool(args.cache)} | '
          f'RAM in use {rss_gb():.1f} GB', flush=True)

    if args.arm == 'latent_adamw':
        n = 2.08e9
        print(f'Full fine-tuning with AdamW keeps, per ternary weight, an fp32 master copy (4 B), two Adam '
              f'moments (8 B) and a gradient (4 B): about {n * 16 / 2**30:.0f} GB for this model\'s ~2.1B '
              f'ternary weights, before activations. It does not fit in 24 GB of RAM or a 16 GB GPU; not run.')
        json.dump(dict(arm='latent_adamw', fits=False, est_gb=n * 16 / 2**30), open(os.path.join(args.out, 'result.json'), 'w'))
        return

    t0 = time.time()
    model, tok, layers = build(args)
    nn.Module.to(model, DEV)
    if args.cache and args.arm != 'latent_sgd':
        for l in layers:
            l.build_cache()
    if args.grad_ckpt and args.arm != 'zeroshot':
        # safe with in-backward updates: a decoder layer is recomputed before its own backward starts,
        # i.e. before any of its weights are updated
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            print('gradient checkpointing on', flush=True)
        except Exception as e:
            print(f'gradient checkpointing unavailable ({type(e).__name__}: {e}); running without it', flush=True)
    n_w = sum(l.dout * l.din for l in layers)
    errs = [getattr(l, 'split_err', 0.0) for l in layers]
    n_vec = sum(torch.is_tensor(getattr(l, 'beta', 0.0)) for l in layers)
    print(f'ternary weights: max relative reconstruction error {max(errs):.2e} (0 = checkpoint already ternary, '
          f'reproduced exactly) | {n_vec} of {len(layers)} layers use per-row/column scales', flush=True)
    state = {'zeroshot': 1, 'counter': 2, 'bop': 3, 'sr': 1, 'topk': 5, 'qat': 16, 'lora': 1, 'tadapt': 1, 'latent_sgd': 4}[args.arm]
    print(f'loaded in {time.time() - t0:.0f}s | {len(layers)} ternary layers, {n_w / 1e9:.2f}B weights | '
          f'training state of the ternary layers ~{n_w * state / 2**30:.1f} GB | memory in use {mem_gb():.1f} GB', flush=True)

    code_info = None
    if args.task in GEN_TASKS:
        train, val, code_info = gen_data(args, tok); label_ids = None
    elif args.task != 'text':
        train, val = cls_data(args)
        label_ids = [label_token(tok, w) for w in TASKS[args.task]['labels']]
        assert len(set(label_ids)) == len(label_ids), 'label words share their first token'
        counts = np.bincount([y for _, y in val], minlength=len(label_ids))
        print(f'task {args.task}: {len(train)} train / {len(val)} eval examples | majority-class accuracy '
              f'{counts.max() / counts.sum():.3f}', flush=True)
    elif args.task == 'text':
        train, val = text_data(args, tok); label_ids = None
    log_f = open(os.path.join(args.out, 'log.jsonl'), 'a')

    if args.gsm_selfgen:
        res = gsm_selfgen(model, tok, args)
        json.dump(res, open(os.path.join(args.out, 'result.json'), 'w'), indent=1)
        return
    if args.arm == 'zeroshot' and args.task in GEN_TASKS:
        res = dict(arm='zeroshot', task='code', data=code_info, eval=dict(dev_loss=code_dev_loss(model, tok, val, args)))
        print('EVAL dev loss', res['eval'], flush=True)
        res['code'] = gen_eval(model, tok, args, 'zeroshot')
        if args.also_eval:
            res['also_eval'] = eval_other_tasks(model, tok, args)
        json.dump(res, open(os.path.join(args.out, 'result.json'), 'w'), indent=1)
        print('RESULT', json.dumps({k: v for k, v in res.items() if k != 'data'}), flush=True)
        return
    if args.arm == 'zeroshot':
        t1 = time.time(); res = evaluate(model, tok, val, label_ids, args)
        res.update(arm='zeroshot', device=str(DEV), eval_seconds=time.time() - t1, peak_mem_gb=mem_gb())
        if args.also_eval:
            res['also_eval'] = eval_other_tasks(model, tok, args)
        print('RESULT', json.dumps(res), flush=True)
        json.dump(res, open(os.path.join(args.out, 'result.json'), 'w'), indent=1)
        return

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0) if args.arm in ('lora', 'qat', 'tadapt') else None
    if args.arm == 'qat' and args.qat_opt == 'adamw8bit':
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(params, lr=args.lr, weight_decay=0.0)
    steps = int(math.ceil(len(train) / args.bs) * args.epochs)
    if args.max_steps:
        steps = min(steps, args.max_steps)
    sched = None
    if opt and args.lr_sched == 'linear':
        warm = max(1, int(0.03 * steps))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: (i + 1) / warm if i < warm else max(0.0, (steps - i) / (steps - warm)))
    print(f'{steps} steps of {args.bs} examples', flush=True)
    model.train()
    peak, tok_count, t_train = mem_gb(), 0, time.time()
    rng = random.Random(args.seed)
    batches = []
    tau_start, tau_hist, tau_trace = args.tau, [], []      # adaptive tau (the layers read args.tau at every update)
    tau_lo = args.tau_min or args.tau
    for step in range(steps):
        if not batches:
            batches = make_batches(train, args.bs, rng)
        batch = [train[i] for i in batches.pop()]
        if args.task in GEN_TASKS:
            loss, k = code_loss(model, batch, tok)
        elif args.task != 'text':
            lg, k = cls_logits(model, tok, batch, label_ids, args)
            loss = F.cross_entropy(lg, torch.tensor([y for _, y in batch], device=DEV))
        else:
            loss, k = text_loss(model, batch)
        loss.backward()                                   # counter / latent_sgd update their layers here
        if args.arm == 'counter' and args.tau_mode == 'adaptive':
            tau_hist.append(loss.item())
            w = args.tau_win
            if len(tau_hist) >= 2 * w and len(tau_hist) % w == 0:
                prev, cur = np.array(tau_hist[-2 * w:-w]), np.array(tau_hist[-w:])
                se = math.sqrt((prev.var(ddof=1) + cur.var(ddof=1)) / w)
                z = (prev.mean() - cur.mean()) / max(se, 1e-8)                 # > 0: the loss went down
                args.tau *= 0.95 if z > 1 else (1.10 if z < -1 else 1.0)
                args.tau = min(args.tau_max, 127.0, max(tau_lo, args.tau))
                tau_trace.append((step + 1, round(args.tau, 2), round(float(z), 2)))
                tau_hist = tau_hist[-w:]
        if opt:
            opt.step(); opt.zero_grad(set_to_none=True)
            if sched: sched.step()
        else:
            for p in params:
                p.grad = None
        tok_count += k
        peak = max(peak, mem_gb())
        el = time.time() - t_train
        flips = sum(l.flips for l in layers) / n_w if args.arm in ('counter', 'bop', 'sr', 'latent_sgd') else None
        rec = dict(step=step + 1, loss=loss.item(), tok_s=tok_count / el, s_per_step=el / (step + 1),
                   elapsed=el, mem_gb=mem_gb(), flips=flips)
        log_f.write(json.dumps(rec) + '\n'); log_f.flush()
        if step < 3 or (step + 1) % 10 == 0 or step + 1 == steps:
            eta = el / (step + 1) * (steps - step - 1)
            print(f"step {step + 1}/{steps} loss {rec['loss']:.4f} | {rec['tok_s']:.1f} tok/s | "
                  f"{rec['s_per_step']:.1f} s/step | ETA {eta / 60:.0f} min | memory {rec['mem_gb']:.1f} GB"
                  + (f' | flipped {flips * 100:.4f}%' if flips is not None else '')
                  + (f' | tau {args.tau:.1f}' if args.tau_mode == 'adaptive' else ''), flush=True)
        if args.eval_every and (step + 1) % args.eval_every == 0 and step + 1 < steps and not args.skip_eval:
            r = evaluate(model, tok, val, label_ids, args)
            log_f.write(json.dumps(dict(step=step + 1, eval=r)) + '\n'); print('EVAL', step + 1, r, flush=True)

    train_s = time.time() - t_train
    if args.arm == 'topk':
        topk_finalize(layers, args.topk_frac)
    if args.tau_mode == 'adaptive':
        print(f'adaptive tau: {tau_start} -> {args.tau:.1f} | trace {tau_trace}', flush=True)
    res = dict(arm=args.arm, device=str(DEV), steps=steps, train_seconds=train_s, s_per_step=train_s / steps,
               tok_s=tok_count / train_s, peak_mem_gb=peak, args={k: v for k, v in vars(args).items()})
    if args.tau_mode == 'adaptive':
        res.update(tau_start=tau_start, tau_end=args.tau, tau_trace=tau_trace)
    if args.arm in ('counter', 'bop', 'sr', 'latent_sgd'):
        res['flips_total_frac'] = sum(l.flips for l in layers) / n_w   # flips counted every step (a weight can flip more than once)
    if args.skip_eval:
        print('RESULT', json.dumps({k: v for k, v in res.items() if k != 'args'}), flush=True)
        json.dump(res, open(os.path.join(args.out, 'result.json'), 'w'), indent=1)
        return
    if args.arm == 'tadapt':
        # the adapter is merged before any evaluation: everything below scores the deployable ternary model
        model.eval()
        with torch.no_grad():
            probe = val[:4]
            if args.task in GEN_TASKS:
                before = code_loss(model, probe, tok)[0].item()
            else:
                before = cls_logits(model, tok, probe, label_ids, args)[0].float()
            n_ad = sum(l.Ka.numel() + l.Kb.numel() for l in layers)
            for l in layers:
                l.merge_tadapt()
            if args.task in GEN_TASKS:
                diff = abs(code_loss(model, probe, tok)[0].item() - before)
            else:
                diff = float((cls_logits(model, tok, probe, label_ids, args)[0].float() - before).abs().max())
        model.train()
        ch = sum(int((l.t.cpu() != l.t_init).sum()) for l in layers)
        zeroed = sum(int(((l.t.cpu() == 0) & (l.t_init != 0)).sum()) for l in layers)
        res.update(adapter_params=n_ad, adapter_mb=n_ad * 4 / 2**20, merge_check_maxdiff=diff,
                   weights_changed=ch, weights_changed_frac=ch / n_w, weights_zeroed_frac=zeroed / n_w)
        print(f'merged the ternary adapter ({n_ad} latent factor entries): {ch / n_w * 100:.4f}% of weights changed '
              f'({zeroed / n_w * 100:.4f}% set to 0) | adapter vs merged output max difference {diff:.2e}', flush=True)
    if args.task in GEN_TASKS:
        res['data'] = code_info
        res['eval'] = dict(dev_loss=code_dev_loss(model, tok, val, args))
        print('EVAL final dev loss', res['eval'], flush=True)
        res['code'] = gen_eval(model, tok, args, 'adapter' if args.arm == 'lora' else 'final')
    else:
        res['eval'] = evaluate(model, tok, val, label_ids, args)
        print('EVAL final', res['eval'], flush=True)
    if args.also_eval:
        res['also_eval'] = eval_other_tasks(model, tok, args)          # LoRA: with the adapter (merged scored below)

    if args.arm == 'lora':
        n_ad = sum(p.numel() for l in layers for p in (l.A, l.B))
        res['adapter_mb'] = n_ad * 4 / 2**20
        for l in layers:
            l.merge_lora(rescale=False)
        res['merge_changed_frac'] = sum(int((l.t.cpu() != l.t0).sum()) for l in layers) / n_w
        if DEV.type == 'cuda':
            torch.cuda.empty_cache()
        if args.task in GEN_TASKS and res['merge_changed_frac'] == 0:
            res['code_merged'] = 'identical to the zero-shot model (no weight changed), generation skipped'
            res['eval_merged_ternary'] = dict(dev_loss=code_dev_loss(model, tok, val, args))
        elif args.task in GEN_TASKS:
            res['eval_merged_ternary'] = dict(dev_loss=code_dev_loss(model, tok, val, args))
            res['code_merged'] = gen_eval(model, tok, args, 'merged')
        else:
            res['eval_merged_ternary'] = evaluate(model, tok, val, label_ids, args)
        if args.also_eval:
            res['also_eval_adapter'] = res.get('also_eval')
            res['also_eval'] = eval_other_tasks(model, tok, args)        # on the deployable merged model
        print(f"EVAL after merging LoRA into ternary weights (scale kept; {res['merge_changed_frac']*100:.4f}% of weights changed)",
              res['eval_merged_ternary'], flush=True)
        if args.task not in GEN_TASKS:
            for l in layers:
                l.merge_lora(rescale=True)
            res['eval_merged_ternary_rescaled'] = evaluate(model, tok, val, label_ids, args)
            print('EVAL after merging with re-ternarization (scale recomputed)', res['eval_merged_ternary_rescaled'], flush=True)
    if args.arm in ('counter', 'bop', 'sr', 'topk', 'qat'):
        patch, n_pos = {}, 0
        for i, l in enumerate(layers):
            if l.mode == 'qat':                            # deployable ternary weights = rounded latent weights
                qs = args.qat_scale
                b = (l.Qs.detach().abs() if qs == 'learned' else
                     l.Wl.detach().float().abs().mean() if qs == 'absmean' else l.beta)
                l.t.copy_((l.Wl.detach().float() / b).round().clamp(-1, 1).to(torch.int8))
                if qs != 'fixed':                          # the deployed layer uses the scale it was trained with
                    if torch.is_tensor(l.beta): l.beta.fill_(float(b))
                    else: l.beta = float(b)
            if l.mode in ('sr', 'topk', 'qat'):
                pos = (l.t.flatten().cpu() != l.t_init.flatten()).nonzero().squeeze(1)
                if not len(pos):
                    continue
            elif l.flipped:
                pos = torch.unique(torch.cat(l.flipped))
            else:
                continue
            patch[f'pos{i}'] = pos.numpy()
            patch[f'val{i}'] = l.t.flatten().cpu()[pos.long()].numpy()
            n_pos += len(pos)
        path = os.path.join(args.out, 'flip_patch.npz')
        np.savez_compressed(path, **patch)
        res.update(weights_changed=n_pos, weights_changed_frac=n_pos / n_w,
                   flip_patch_mb=os.path.getsize(path) / 2**20)
        print(f'{n_pos} weights changed ({n_pos / n_w * 100:.4f}%), flip patch {res["flip_patch_mb"]:.1f} MB', flush=True)

    print('RESULT', json.dumps({k: v for k, v in res.items() if k != 'args'}), flush=True)
    json.dump(res, open(os.path.join(args.out, 'result.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
