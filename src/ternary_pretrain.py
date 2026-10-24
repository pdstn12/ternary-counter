#!/usr/bin/env python3
"""Pretrain a small LLaMA-style model from scratch with ternary linear layers, comparing update rules.

  fp       full-precision reference (bf16 matmuls, fp32 weights + AdamW): not ternary
  qat      the standard ternary recipe (BitNet b1.58): fp32 latent weights + AdamW, absmean ternarization in the
           forward pass, straight-through estimator; ~16 bytes of training state per weight
  counter  ours: ternary weights + an int8 counter per weight, no latent weights, no optimizer state for them;
           ~2 bytes per weight
  sr       DQT-style stochastic rounding straight onto the ternary grid, no memory; ~1 byte per weight

All arms share the data, model shape, token budget, schedule shape and 8-bit activation quantization (ternary arms).
Embeddings, norms and (ternary arms) one scale per matrix are small fp32 tensors trained with AdamW in every arm.

  python ternary_pretrain.py --arm counter --data /path/tinystories --out runs/counter --tokens 2e8
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', required=True, choices=['fp', 'qat', 'counter', 'sr'])
    p.add_argument('--data', required=True, help='folder with train.bin / val.bin (uint16 GPT-2 tokens)')
    p.add_argument('--out', required=True)
    p.add_argument('--d', type=int, default=512)
    p.add_argument('--layers', type=int, default=8)
    p.add_argument('--heads', type=int, default=8)
    p.add_argument('--ffn', type=int, default=1408)
    p.add_argument('--seq', type=int, default=512)
    p.add_argument('--bs', type=int, default=64, help='sequences per optimizer step')
    p.add_argument('--micro', type=int, default=16, help='sequences per forward pass (gradient accumulation; keeps the 50k-vocab logits small)')
    p.add_argument('--tokens', type=float, default=2e8, help='training tokens')
    p.add_argument('--lr', type=float, default=1.5e-3, help='AdamW lr (all weights for fp/qat; small fp tensors otherwise)')
    p.add_argument('--wd', type=float, default=0.1, help='weight decay on matrices (fp/qat)')
    p.add_argument('--warmup', type=float, default=0.01, help='fraction of steps; then linear decay to 10%%')
    p.add_argument('--vote', type=float, default=0.3, help='counter: votes per step at unit normalized gradient')
    p.add_argument('--tau', type=int, default=8, help='counter: flip threshold (the starting value when adaptive)')
    p.add_argument('--tau_end', type=float, default=26, help='linear: tau at the last step (tau -> tau_end)')
    p.add_argument('--tau_mode', default='fixed', choices=['fixed', 'adaptive', 'linear'],
                   help='adaptive: every --tau_win steps compare the mean training loss of the last window with the one '
                        'before; clearly better -> tau x0.95 (flip more easily), no clear change -> unchanged, clearly worse -> x1.10')
    p.add_argument('--tau_win', type=int, default=20)
    p.add_argument('--tau_min', type=float, default=4)
    p.add_argument('--tau_max', type=float, default=64)
    p.add_argument('--leak', type=float, default=0.0, help='counter: stochastic decay of each counter toward 0')
    p.add_argument('--sr_step', type=float, default=0.01, help='sr: move probability at unit normalized gradient')
    p.add_argument('--clip', type=float, default=1.0)
    p.add_argument('--eval_every', type=float, default=1e7, help='evaluate every N training tokens')
    p.add_argument('--eval_tokens', type=float, default=1e6)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


# ----------------------------------------------------------------------------------------------- layers
def act_quant(x):
    """Per-token absmax int8 activations (BitNet b1.58), straight-through."""
    s = 127.0 / x.detach().abs().amax(-1, keepdim=True).clamp(min=1e-5)
    return x + ((x.detach() * s).round().clamp(-128, 127) / s - x.detach())


class TernaryMatmul(torch.autograd.Function):
    """y = x @ t^T with t in {-1,0,1} (int8 buffer, no gradient). Backward returns dL/dx and leaves dL/dt on the
    layer for its integer update rule; dL/dt is never accumulated across steps."""
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(ctx, x, layer):
        ctx.layer = layer
        ctx.save_for_backward(x)
        return F.linear(x, layer.t.to(x.dtype))

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, go):
        (x,) = ctx.saved_tensors
        layer = ctx.layer
        gx = go @ layer.t.to(go.dtype)
        g2, x2 = go.reshape(-1, go.shape[-1]), x.reshape(-1, x.shape[-1])
        gW = (g2.t() @ x2).float()
        layer.gW = gW if layer.gW is None else layer.gW + gW
        return gx, None


class TLinear(nn.Module):
    def __init__(self, din, dout, arm, args):
        super().__init__()
        self.arm, self.args = arm, args
        W = torch.randn(dout, din) * 0.02
        if arm in ('fp', 'qat'):
            self.weight = nn.Parameter(W)
        else:
            beta = W.abs().mean()
            self.register_buffer('t', (W / beta).round().clamp(-1, 1).to(torch.int8))
            self.scale = nn.Parameter(beta.clone())         # one fp32 scale per matrix, trained with AdamW
            if arm == 'counter':
                self.register_buffer('c', torch.zeros(dout, din, dtype=torch.int8))
            self.gW, self.flips = None, 0

    def forward(self, x):
        if self.arm == 'fp':
            return F.linear(x, self.weight)
        x = act_quant(x)
        if self.arm == 'qat':
            W = self.weight
            b = W.detach().abs().mean().clamp(min=1e-5)
            Wq = (W / b).round().clamp(-1, 1) * b
            return F.linear(x, W + (Wq - W).detach())
        return TernaryMatmul.apply(x, self) * self.scale

    @torch.no_grad()
    def update(self, strength, tau=None):
        """strength: the scheduled vote (counter) or move probability (sr) at unit normalized gradient."""
        g, self.gW = self.gW, None
        if g is None:
            return
        s = -g / g.pow(2).mean().sqrt().clamp(min=1e-30)
        t = self.t.float()
        if self.arm == 'counter':
            a = self.args
            cf = self.c.float()
            if a.leak:
                cf = cf - torch.sign(cf) * (torch.rand_like(cf) < (a.leak * cf.abs()).clamp(max=1)).float()
            cf = (cf + torch.floor(strength * s + torch.rand_like(s))).clamp(-127, 127)
            tau = a.tau if tau is None else tau
            up, dn = cf >= tau, cf <= -tau
            nt = torch.where(up, (t + 1).clamp(max=1), torch.where(dn, (t - 1).clamp(min=-1), t))
            cf = torch.where(up | dn, torch.zeros_like(cf), cf)
            self.c.copy_(cf.to(torch.int8))
        else:                                                # sr
            d = strength * s
            move = torch.rand_like(d) < d.abs().clamp(max=1)
            nt = torch.where(move & (d > 0), (t + 1).clamp(max=1), torch.where(move & (d < 0), (t - 1).clamp(min=-1), t))
        self.flips += int((nt != t).sum())
        self.t.copy_(nt.to(torch.int8))


# ----------------------------------------------------------------------------------------------- model
class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(x.dtype) * self.w


def rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), -1).flatten(-2)


class Block(nn.Module):
    def __init__(self, a):
        super().__init__()
        d, h, f = a.d, a.heads, a.ffn
        self.h = h
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.q, self.k, self.v, self.o = (TLinear(d, d, a.arm, a) for _ in range(4))
        self.gate, self.up, self.down = TLinear(d, f, a.arm, a), TLinear(d, f, a.arm, a), TLinear(f, d, a.arm, a)

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        y = self.n1(x)
        q, k, v = (m(y).view(B, T, self.h, D // self.h).transpose(1, 2) for m in (self.q, self.k, self.v))
        q, k = rope(q, cos, sin), rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, D)
        x = x + self.o(y)
        y = self.n2(x)
        return x + self.down(F.silu(self.gate(y)) * self.up(y))


class LM(nn.Module):
    def __init__(self, a, vocab=50304):
        super().__init__()
        self.emb = nn.Embedding(vocab, a.d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.blocks = nn.ModuleList(Block(a) for _ in range(a.layers))
        self.norm = RMSNorm(a.d)
        hd = a.d // a.heads
        inv = 1.0 / (10000 ** (torch.arange(0, hd, 2).float() / hd))
        ang = torch.outer(torch.arange(a.seq).float(), inv)
        self.register_buffer('cos', ang.cos()[None, None], persistent=False)
        self.register_buffer('sin', ang.sin()[None, None], persistent=False)

    def forward(self, idx, targets=None):
        T = idx.shape[1]
        x = self.emb(idx)
        cos, sin = self.cos[:, :, :T], self.sin[:, :, :T]
        for b in self.blocks:
            x = b(x, cos, sin)
        logits = F.linear(self.norm(x), self.emb.weight)       # tied, full precision (as in BitNet)
        if targets is None:
            return logits
        return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1))


# ----------------------------------------------------------------------------------------------- training
def main():
    a = get_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device('cuda')
    train = np.memmap(os.path.join(a.data, 'train.bin'), dtype=np.uint16, mode='r')
    val = np.memmap(os.path.join(a.data, 'val.bin'), dtype=np.uint16, mode='r')
    model = LM(a).to(dev)
    tl = [m for m in model.modules() if isinstance(m, TLinear)]
    n_t = sum(m.t.numel() if hasattr(m, 't') else m.weight.numel() for m in tl)
    n_all = sum(p.numel() for p in model.parameters()) + (sum(m.t.numel() for m in tl if hasattr(m, 't')))
    mats = [p for n, p in model.named_parameters() if p.dim() == 2 and 'emb' not in n]
    rest = [p for n, p in model.named_parameters() if not (p.dim() == 2 and 'emb' not in n)]
    groups = [g for g in ({'params': mats, 'weight_decay': a.wd}, {'params': rest, 'weight_decay': 0.0}) if g['params']]
    opt = torch.optim.AdamW(groups, lr=a.lr, betas=(0.9, 0.95))
    accum = max(1, a.bs // a.micro)
    tok_step = a.bs * a.seq
    steps = int(a.tokens // tok_step)
    warm = max(1, int(a.warmup * steps))
    sched = lambda i: (i + 1) / warm if i < warm else 1.0 - 0.9 * (i - warm) / max(1, steps - warm)
    state_b = {'fp': 16, 'qat': 16, 'counter': 2, 'sr': 1}[a.arm]
    print(f'arm {a.arm} | {n_t / 1e6:.1f}M weights in linear layers ({"ternary" if a.arm != "fp" else "fp"}), '
          f'{n_all / 1e6:.1f}M in total | training state of the linear layers ~{state_b} B/weight = '
          f'{n_t * state_b / 2**20:.0f} MB | {steps} steps of {tok_step} tokens', flush=True)

    rng = np.random.default_rng(a.seed)
    def batch(data, n):
        ix = rng.integers(0, len(data) - a.seq - 1, n)
        x = np.stack([data[i:i + a.seq + 1].astype(np.int64) for i in ix])
        x = torch.from_numpy(x).pin_memory().to(dev, non_blocking=True)
        return x[:, :-1], x[:, 1:]

    n_eval = int(a.eval_tokens // a.seq)
    ev_ix = np.arange(n_eval) * ((len(val) - a.seq - 1) // n_eval)     # fixed, spread over the val split
    @torch.no_grad()
    def evaluate():
        model.eval(); tot = 0.0
        for i in range(0, n_eval, a.micro):
            x = np.stack([val[j:j + a.seq + 1].astype(np.int64) for j in ev_ix[i:i + a.micro]])
            x = torch.from_numpy(x).to(dev)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                tot += model(x[:, :-1], x[:, 1:]).item() * len(x)
        model.train()
        return tot / n_eval

    log = open(os.path.join(a.out, 'log.jsonl'), 'w')
    curve, t0, next_eval = [], time.time(), a.eval_every
    tau_now, hist, tau_trace = float(a.tau), [], []          # adaptive tau: training-loss history per step
    torch.cuda.reset_peak_memory_stats()
    for step in range(steps):
        f = sched(step)
        for g in opt.param_groups:
            g['lr'] = a.lr * f
        step_loss = 0.0
        for _ in range(accum):                               # counter / sr: dL/dt accumulates on each layer
            x, y = batch(train, a.micro)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = model(x, y)
            (loss / accum).backward()
            step_loss += loss.item() / accum
        hist.append(step_loss)
        if a.arm == 'counter' and a.tau_mode == 'adaptive' and len(hist) >= 2 * a.tau_win and len(hist) % a.tau_win == 0:
            prev, cur = np.array(hist[-2 * a.tau_win:-a.tau_win]), np.array(hist[-a.tau_win:])
            se = math.sqrt((prev.var(ddof=1) + cur.var(ddof=1)) / a.tau_win)   # noise of the difference of the two means
            z = (prev.mean() - cur.mean()) / max(se, 1e-8)                     # > 0: the loss went down
            tau_now *= 0.95 if z > 1 else (1.10 if z < -1 else 1.0)
            tau_now = min(a.tau_max, max(a.tau_min, tau_now))
            tau_trace.append((step + 1, round(tau_now, 3), round(float(z), 2)))
            hist = hist[-a.tau_win:]
        if a.clip:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.grad is not None], a.clip)
        opt.step(); opt.zero_grad(set_to_none=True)
        if a.tau_mode == 'linear':                           # control: tau rises on a fixed schedule, no feedback
            tau_now = a.tau + (a.tau_end - a.tau) * step / max(1, steps - 1)
        if a.arm in ('counter', 'sr'):
            strength = (a.vote if a.arm == 'counter' else a.sr_step) * f
            for m in tl:
                m.update(strength, int(round(tau_now)) if a.tau_mode != 'fixed' else None)
        done = (step + 1) * tok_step
        if step % 50 == 0 or step + 1 == steps:
            el = time.time() - t0
            fl = sum(m.flips for m in tl) / n_t if a.arm in ('counter', 'sr') else None
            print(f'step {step + 1}/{steps} loss {loss.item():.3f} | {done / el / 1e3:.0f}k tok/s | ETA {(steps - step - 1) * el / (step + 1) / 60:.0f} min'
                  + (f' | flips so far {100 * fl:.2f}% of weights' if fl is not None else '')
                  + (f' | tau {tau_now:.1f}' if a.tau_mode != 'fixed' else ''), flush=True)
            log.write(json.dumps(dict(step=step + 1, tokens=done, loss=loss.item(), flips=fl)) + '\n'); log.flush()
        if done >= next_eval or step + 1 == steps:
            vl = evaluate()
            curve.append((done, vl))
            print(f'EVAL tokens {done / 1e6:.0f}M val loss {vl:.4f} ppl {math.exp(vl):.2f}', flush=True)
            log.write(json.dumps(dict(tokens=done, val_loss=vl)) + '\n'); log.flush()
            next_eval += a.eval_every
            if not math.isfinite(vl):
                print('diverged: stopping', flush=True); break
    el = time.time() - t0
    # a short greedy sample from the trained (ternary) model
    model.eval()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('gpt2')
    ids = torch.tensor([tok.encode('Once upon a time')], device=dev)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for _ in range(60):
            ids = torch.cat([ids, model(ids[:, -a.seq:])[:, -1:].argmax(-1)], 1)
    sample = tok.decode(ids[0])
    vl = curve[-1][1]
    res = dict(arm=a.arm, args=vars(a), steps=step + 1, tokens=done, val_loss=vl, val_ppl=math.exp(vl) if math.isfinite(vl) else None,
               curve=curve, seconds=el, tok_per_s=done / el, peak_mem_gb=torch.cuda.max_memory_allocated() / 2**30,
               state_bytes_per_weight=state_b, linear_weights=n_t, total_params=n_all,
               flips_frac=(sum(m.flips for m in tl) / n_t if a.arm in ('counter', 'sr') else None), sample=sample,
               tau_trace=tau_trace)
    print('SAMPLE:', sample.replace('\n', ' '), flush=True)
    print('RESULT', json.dumps({k: v for k, v in res.items() if k not in ('args', 'curve', 'sample')}), flush=True)
    json.dump(res, open(os.path.join(a.out, 'result.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
