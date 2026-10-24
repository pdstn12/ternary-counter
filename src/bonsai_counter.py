"""Int8-counter fine-tuning of Ternary Bonsai 2 (PrismML, Qwen3.5 hybrid architecture) straight from its PQ2_0 GGUF.

The model is a PyTorch port of the graph in PrismML's llama.cpp fork (src/models/qwen35.cpp, branch `prism`):
- weights stay ternary: 2-bit codes (-1/0/+1) packed 4 per byte exactly as in PQ2_0, plus the fp16 scale of each
  group of 128 weights, which is kept fixed;
- Hadamard-folded weights: y = W_stored @ H_b(signs * perm(x)), H_b = block-diagonal normalized Sylvester Hadamard;
- each trainable layer has an int8 counter; the update runs inside the layer's backward pass, so no gradient is
  ever stored for the whole model (state = 1 B counter + 0.25 B packed code per weight);
- after training the codes are written back into a copy of the GGUF (same block layout), so the result runs in
  PrismML's llama.cpp on a CPU.
"""
import argparse, json, math, os, re, shutil, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

PQ2_0 = 142            # ggml type id in PrismML's fork
QK = 128               # weights per PQ2_0 block
SHIFTS = None


# ----------------------------------------------------------------------------- GGUF helpers
def fval(reader, key, default=None):
    f = reader.fields.get(key)
    if f is None:
        return default
    v = f.contents()
    return v


def unpack(packed):
    """uint8 [out, in/4] -> int8 codes [out, in] in {-1, 0, +1} (PQ2_0 bit order: element j -> byte j//4, bits 2*(j%4))."""
    global SHIFTS
    if SHIFTS is None or SHIFTS.device != packed.device:
        SHIFTS = torch.tensor([0, 2, 4, 6], dtype=torch.uint8, device=packed.device)
    q = (packed.unsqueeze(-1) >> SHIFTS) & 3
    return q.reshape(packed.shape[0], -1).to(torch.int8) - 1


def pack(codes):
    v = (codes + 1).to(torch.uint8).reshape(codes.shape[0], -1, 4)
    return v[..., 0] | (v[..., 1] << 2) | (v[..., 2] << 4) | (v[..., 3] << 6)


def pq2_split(raw, out_f, in_f):
    """raw block bytes [out, in/128 * 34] -> (packed uint8 [out, in/4], scales fp16 [out, in/128])."""
    b = raw.reshape(out_f, in_f // QK, 2 + QK // 4)
    scales = np.ascontiguousarray(b[..., :2]).view(np.float16).reshape(out_f, in_f // QK)
    packed = np.ascontiguousarray(b[..., 2:]).reshape(out_f, in_f // 4)
    return packed, scales


def hadamard(b, device):
    i = torch.arange(b, device=device)
    x = i[:, None] & i[None, :]
    par = torch.zeros_like(x)
    while bool(x.any()):
        par ^= x & 1
        x = x >> 1
    return (1 - 2 * par).float() / math.sqrt(b)


class Rot:
    """Input-side transform of a Hadamard-folded weight: H_b(signs * perm(x)) (signs then H, as build_lora_mm)."""
    def __init__(self, H, signs=None, perm=None, inverse=False):
        self.H, self.signs, self.perm, self.inverse = H, signs, perm, inverse

    def __call__(self, x):
        dt, shp, b = x.dtype, x.shape, self.H.shape[0]
        x = x.float()
        if self.inverse:                         # embedding rows: h = signs * (H z)
            x = (x.reshape(*shp[:-1], -1, b) @ self.H).reshape(shp)
            return (x * self.signs if self.signs is not None else x).to(dt)
        if self.perm is not None:                # tiled [hd, nk, rep] -> grouped [hd, rep, nk] (ggml order)
            hd, nk, rep = self.perm
            x = x.reshape(*shp[:-1], rep, nk, hd).transpose(-3, -2).reshape(shp)
        if self.signs is not None:
            x = x * self.signs
        return (x.reshape(*shp[:-1], -1, b) @ self.H).reshape(shp).to(dt)


# ----------------------------------------------------------------------------- counter-trained ternary layer
class TernFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, anchor, mod):
        ctx.mod = mod
        ctx.save_for_backward(x)
        return x @ mod.weight(x.dtype).t()

    @staticmethod
    def backward(ctx, gy):
        (x,) = ctx.saved_tensors
        mod = ctx.mod
        gx = gy @ mod.weight(gy.dtype)
        if mod.trainable and mod.voting:
            g2, x2 = gy.reshape(-1, gy.shape[-1]), x.reshape(-1, x.shape[-1])
            mod.vote(g2.t().matmul(x2).float())
        return gx, torch.zeros_like(mod.anchor), None


class TLinear(nn.Module):
    def __init__(self, name, packed, scales, rot, trainable, hp):
        super().__init__()
        self.name, self.rot, self.trainable, self.hp, self.voting = name, rot, trainable, hp, False
        self.out_f, self.in_f = packed.shape[0], packed.shape[1] * 4
        self.register_buffer('packed', packed)
        self.register_buffer('scales', scales)
        self.counter = torch.zeros(self.out_f, self.in_f, dtype=torch.int8, device=packed.device) if trainable else None
        # a scalar that requires grad, so autograd reaches every layer's backward (the weights are not Parameters)
        self.anchor = nn.Parameter(torch.zeros((), device=packed.device), requires_grad=trainable)
        self.flips = 0

    def weight(self, dtype):
        c = unpack(self.packed).to(dtype).reshape(self.out_f, -1, QK)
        return (c * self.scales.to(dtype).unsqueeze(-1)).reshape(self.out_f, self.in_f)

    def forward(self, x, xr=None):
        if self.rot is not None:
            x = xr if xr is not None else self.rot(x)
        return TernFn.apply(x, self.anchor, self)

    @torch.no_grad()
    def vote(self, gW):                          # identical to the counter rule of bitnet_cpu_finetune.py
        hp = self.hp
        s = -hp['vote'] * gW / gW.pow(2).mean().sqrt().clamp(min=1e-30)
        cf = self.counter.float()
        if hp['leak']:
            cf = cf - torch.sign(cf) * (torch.rand_like(cf) < (hp['leak'] * cf.abs()).clamp(max=1)).float()
        cf = (cf + torch.floor(s + torch.rand_like(s))).clamp(-127, 127)
        del s
        up, dn = cf >= hp['tau'], cf <= -hp['tau']
        if bool((up | dn).any()):
            t = unpack(self.packed)
            nt = torch.where(up, (t + 1).clamp(max=1), torch.where(dn, (t - 1).clamp(min=-1), t))
            k = int((nt != t).sum())
            if k:
                self.packed.copy_(pack(nt)); self.flips += k
            cf = torch.where(up | dn, torch.zeros_like(cf), cf)
        self.counter.copy_(cf.to(torch.int8))


class Dense(nn.Module):                         # frozen non-ternary weight (F32/F16/BF16 or other quant, dequantized)
    def __init__(self, W, rot):
        super().__init__()
        self.register_buffer('W', W)
        self.rot = rot

    def forward(self, x, xr=None):
        if self.rot is not None:
            x = xr if xr is not None else self.rot(x)
        return x @ self.W.to(x.dtype).t()


def rms(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(x.dtype)


# ----------------------------------------------------------------------------- gated delta rule kernels
def gdn_naive(q, k, v, g, beta, scale):
    """Reference recurrence of llama.cpp build_delta_net_autoregressive. q,k,v [B,T,H,D], g,beta [B,T,H]."""
    B, T, H, Dk = q.shape
    Dv = v.shape[-1]
    S = torch.zeros(B, H, Dv, Dk, device=q.device, dtype=torch.float32)
    out = []
    q, k, v, g, beta = q.float() * scale, k.float(), v.float(), g.float(), beta.float()
    for t in range(T):
        S = S * g[:, t].exp()[..., None, None]
        sk = torch.einsum('bhvk,bhk->bhv', S, k[:, t])
        d = (v[:, t] - sk) * beta[:, t][..., None]
        S = S + torch.einsum('bhv,bhk->bhvk', d, k[:, t])
        out.append(torch.einsum('bhvk,bhk->bhv', S, q[:, t]))
    return torch.stack(out, 1)


GDN_IMPL = None


def gdn(q, k, v, g, beta, scale):
    if GDN_IMPL == 'fla':
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        dt = q.dtype
        lo = torch.bfloat16 if dt == torch.float32 else dt          # the Triton kernels want 16-bit q, k, v
        o, _ = chunk_gated_delta_rule(q.to(lo), k.to(lo), v.to(lo), g.float(), beta.to(lo), scale=scale)
        return o.to(dt)
    if GDN_IMPL == 'hf':
        from transformers.models.qwen3_next.modeling_qwen3_next import torch_chunk_gated_delta_rule
        o, _ = torch_chunk_gated_delta_rule(q * (scale * math.sqrt(q.shape[-1])), k, v, g.float(), beta.float())
        return o.to(q.dtype)
    return gdn_naive(q, k, v, g, beta, scale).to(q.dtype)


def pick_gdn(device):
    """Choose the fastest kernel that matches the reference recurrence on random inputs."""
    global GDN_IMPL
    torch.manual_seed(0)
    B, T, H, D = 1, 96, 4, 64
    q = F.normalize(torch.randn(B, T, H, D, device=device), dim=-1).bfloat16()
    k = F.normalize(torch.randn(B, T, H, D, device=device), dim=-1).bfloat16()
    v = torch.randn(B, T, H, D, device=device).bfloat16()
    g = -F.softplus(torch.randn(B, T, H, device=device))
    beta = torch.rand(B, T, H, device=device).bfloat16()
    ref = gdn_naive(q, k, v, g, beta, D ** -0.5)
    report = {}
    for impl in ('fla', 'hf'):
        GDN_IMPL = impl
        try:
            o = gdn(q, k, v, g, beta, D ** -0.5).float()
            err = float((o - ref).abs().max() / ref.abs().max())
            report[impl] = round(err, 4)
            if err < 0.03:
                print(f'gated delta rule: using {impl} (max rel err {err:.4f} vs reference)', flush=True)
                return report
        except Exception as e:
            report[impl] = f'failed: {type(e).__name__}: {str(e)[:120]}'
    GDN_IMPL = 'naive'
    print('gated delta rule: WARNING, only the slow reference loop works:', report, flush=True)
    return report


# ----------------------------------------------------------------------------- the model
class Bonsai(nn.Module):
    def __init__(self, gguf_path, device='cuda', hp=None, dtype=torch.bfloat16):
        super().__init__()
        from gguf import GGUFReader
        from gguf.quants import dequantize
        self.hp = hp or dict(tau=32.0, vote=0.1, leak=0.003)
        r = GGUFReader(gguf_path)
        self.path, self.device = gguf_path, device
        arch = fval(r, 'general.architecture')
        assert arch in ('qwen35',), f'unsupported architecture {arch}'
        A = lambda k, d=None: fval(r, f'{arch}.{k}', d)
        self.n_embd, self.n_layer = int(A('embedding_length')), int(A('block_count'))
        self.n_head, self.n_kv = int(A('attention.head_count')), int(A('attention.head_count_kv'))
        self.hd = int(A('attention.key_length'))
        self.eps = float(A('attention.layer_norm_rms_epsilon'))
        self.n_rot, self.base = int(A('rope.dimension_count')), float(A('rope.freq_base'))
        self.sections = [int(s) for s in (A('rope.dimension_sections') or [0, 0, 0, 0])][:4]
        self.d_conv, self.d_inner = int(A('ssm.conv_kernel')), int(A('ssm.inner_size'))
        self.hk, self.nv, self.nk = int(A('ssm.state_size')), int(A('ssm.time_step_rank')), int(A('ssm.group_count'))
        self.hv = self.d_inner // self.nv
        # Hadamard contract
        hv = fval(r, 'prism.hadamard.version')
        self.had = None
        if hv is not None:
            b = int(fval(r, 'prism.hadamard.block_size'))
            assert fval(r, 'prism.hadamard.transform') == 'normalized-sylvester-walsh-hadamard'
            assert fval(r, 'prism.hadamard.axis') == 'input-last-dimension'
            H = hadamard(b, device)
            signs = {}
            if fval(r, 'prism.hadamard.sign_mode') == 'explicit':
                widths, vals, off = list(fval(r, 'prism.hadamard.sign_widths')), list(fval(r, 'prism.hadamard.sign_values')), 0
                for w in widths:
                    signs[int(w)] = torch.tensor(vals[off:off + w], dtype=torch.float32, device=device); off += w
            self.had = dict(version=int(hv), H=H, signs=signs, names=set(fval(r, 'prism.hadamard.weight_names')),
                            inv=set(fval(r, 'prism.hadamard.inverse_weight_names') or []),
                            tied=bool(fval(r, 'prism.hadamard.tied_output', False)),
                            gdn_grouped=bool(fval(r, 'prism.hadamard.gdn_v_grouped', False)))
        self._rots = {}
        T = {t.name: t for t in r.tensors}
        self.recr = [f'blk.{i}.ssm_a' in T for i in range(self.n_layer)]      # linear-attention (GDN) layers
        self.n_tern = self.n_train = 0

        def rot_for(name, width, perm=None, inverse=False):
            h = self.had
            if h is None or (not inverse and name not in h['names']) or (inverse and name not in h['inv']):
                return None
            key = (width, perm, inverse)
            if key not in self._rots:
                self._rots[key] = Rot(h['H'], h['signs'].get(width) if h['signs'] else None, perm, inverse)
                assert not h['signs'] or width in h['signs'], f'no sign vector for width {width}'
            return self._rots[key]

        def dense_np(t):
            ne = [int(x) for x in t.shape]
            if t.tensor_type in (0, 1):          # F32, F16
                a = np.array(t.data, dtype=np.float32)
            elif t.tensor_type == 30:            # BF16
                a = (np.array(t.data).view(np.uint16).astype(np.uint32) << 16).view(np.float32)
            else:
                a = dequantize(t.data, t.tensor_type)
            return a.reshape(list(reversed(ne)))

        def lin(name, perm=None, trainable=True):
            t = T[name]
            out_f, in_f = int(t.shape[1]), int(t.shape[0])
            rot = rot_for(name, in_f, perm)
            if int(t.tensor_type) == PQ2_0:
                pk, sc = pq2_split(np.array(t.data), out_f, in_f)
                pk = torch.from_numpy(pk).to(device)
                assert not bool(((pk & 0x03) == 3).any() | ((pk & 0x0C) == 12).any() | ((pk & 0x30) == 48).any()
                                | ((pk & 0xC0) == 192).any()), f'{name}: code 11 (+2) present, not ternary'
                m = TLinear(name, pk, torch.from_numpy(sc).to(device), rot, trainable, self.hp)
                self.n_tern += out_f * in_f
                self.n_train += out_f * in_f if trainable else 0
                return m
            return Dense(torch.from_numpy(dense_np(t)).to(device, dtype), rot)

        def vec(name):
            return torch.from_numpy(dense_np(T[name]).astype(np.float32)).to(device)

        # embedding (frozen by default) and output head
        te = T['token_embd.weight']
        self.vocab = int(te.shape[1])
        if int(te.tensor_type) == PQ2_0:
            pk, sc = pq2_split(np.array(te.data), self.vocab, self.n_embd)
            E = TLinear('token_embd.weight', torch.from_numpy(pk).to(device), torch.from_numpy(sc).to(device), None, False, self.hp)
            self.E = E.weight(dtype)
            self.n_tern += self.vocab * self.n_embd
            del E
        else:
            self.E = torch.from_numpy(dense_np(te)).to(device, dtype)
        self.embed_rot = rot_for('token_embd.weight', self.n_embd, inverse=True)
        if 'output.weight' in T:
            self.head = lin('output.weight', trainable=False)
        else:
            self.head = None
            tied = self.had is not None and self.had['tied']
            self.head_rot = Rot(self.had['H'], self.had['signs'].get(self.n_embd) if self.had['signs'] else None) if tied else None
        self.out_norm = vec('output_norm.weight')
        self.layers = []
        for i in range(self.n_layer):
            p = f'blk.{i}.'
            L = dict(recr=self.recr[i], attn_norm=vec(p + 'attn_norm.weight'), post_norm=vec(p + 'post_attention_norm.weight'),
                     ffn_gate=lin(p + 'ffn_gate.weight'), ffn_up=lin(p + 'ffn_up.weight'), ffn_down=lin(p + 'ffn_down.weight'))
            if self.recr[i]:
                perm = (self.hv, self.nk, self.nv // self.nk) if (self.had and self.had['gdn_grouped']) else None
                L.update(wqkv=lin(p + 'attn_qkv.weight'), wgate=lin(p + 'attn_gate.weight'),
                         beta=lin(p + 'ssm_beta.weight'), alpha=lin(p + 'ssm_alpha.weight'),
                         conv=vec(p + 'ssm_conv1d.weight'), dt=vec(p + 'ssm_dt.bias'), a=vec(p + 'ssm_a'),
                         ssm_norm=vec(p + 'ssm_norm.weight'), ssm_out=lin(p + 'ssm_out.weight', perm=perm))
            else:
                if p + 'attn_q.weight' in T:
                    L.update(wq=lin(p + 'attn_q.weight'), wk=lin(p + 'attn_k.weight'), wv=lin(p + 'attn_v.weight'))
                else:
                    L.update(wqkv=lin(p + 'attn_qkv.weight'))
                L.update(wo=lin(p + 'attn_output.weight'), q_norm=vec(p + 'attn_q_norm.weight'), k_norm=vec(p + 'attn_k_norm.weight'))
            self.layers.append(L)
            for k, m in L.items():
                if isinstance(m, nn.Module):
                    self.add_module(f'l{i}_{k}', m)
        del r
        self.tlayers = [m for m in self.modules() if isinstance(m, TLinear) and m.trainable]
        self._rope = {}

    # -- pieces
    def rope_cs(self, T):
        if self._rope.get('T', 0) < T:
            T0 = T
            T = max(T, 4096)
            n2 = self.n_rot // 2
            sec = self.sections
            sd = sum(sec)
            comp = []                            # which position component each rotated pair uses (ggml_mrope_cache_init)
            for i in range(n2):
                if sd == 0:
                    comp.append(0); continue
                s = i % sd
                if s % 3 == 1 and s < 3 * sec[1]: comp.append(1)
                elif s % 3 == 2 and s < 3 * sec[2]: comp.append(2)
                elif s % 3 == 0 and s < 3 * sec[0]: comp.append(0)
                else: comp.append(3)
            pos = torch.arange(T, device=self.device, dtype=torch.float32)
            pmat = torch.stack([pos, pos, pos, torch.zeros_like(pos)], 1)       # text: t = h = w = position, e = 0
            p = pmat[:, comp]                                                    # [T, n2]
            inv = self.base ** (-torch.arange(n2, device=self.device, dtype=torch.float32) * 2 / self.n_rot)
            ang = p * inv
            self._rope = dict(T=T, cs=(ang.cos(), ang.sin()))
            T = T0
        c, s = self._rope['cs']
        return c[:T], s[:T]

    def rope(self, x):                           # x [T, H, D], NEOX pairs (i, i + n_rot/2) on the first n_rot dims
        T = x.shape[0]
        cos, sin = self.rope_cs(T)
        n2 = self.n_rot // 2
        xf = x.float()
        x1, x2 = xf[..., :n2], xf[..., n2:self.n_rot]
        c, s = cos[:, None, :], sin[:, None, :]
        out = torch.cat([x1 * c - x2 * s, x2 * c + x1 * s, xf[..., self.n_rot:]], -1)
        return out.to(x.dtype)

    @staticmethod
    def shared_rot(x, mods):
        rots = [m.rot for m in mods if m.rot is not None]
        if rots and all(r is rots[0] for r in rots):
            return rots[0](x)
        return None

    def attn(self, L, x):
        T = x.shape[0]
        if 'wq' in L:
            xr = self.shared_rot(x, [L['wq'], L['wk'], L['wv']])
            qg, k, v = L['wq'](x, xr), L['wk'](x, xr), L['wv'](x, xr)
        else:
            qkv = L['wqkv'](x)
            nq = self.n_head * self.hd * 2
            qg, k, v = qkv[:, :nq], qkv[:, nq:nq + self.n_kv * self.hd], qkv[:, nq + self.n_kv * self.hd:]
        qg = qg.view(T, self.n_head, 2 * self.hd)
        q, gate = qg[..., :self.hd], qg[..., self.hd:].reshape(T, self.n_head * self.hd)
        q = self.rope(rms(q, L['q_norm'], self.eps))
        k = self.rope(rms(k.view(T, self.n_kv, self.hd), L['k_norm'], self.eps))
        v = v.view(T, self.n_kv, self.hd)
        rep = self.n_head // self.n_kv
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        o = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None],
                                           is_causal=True, scale=self.hd ** -0.5)
        o = o[0].transpose(0, 1).reshape(T, self.n_head * self.hd)
        return L['wo'](o * torch.sigmoid(gate))

    def gdn_layer(self, L, x):
        T = x.shape[0]
        xr = self.shared_rot(x, [L['wqkv'], L['wgate']])
        qkv, z = L['wqkv'](x, xr), L['wgate'](x, xr)
        beta = torch.sigmoid(L['beta'](x).float())
        g = F.softplus(L['alpha'](x).float() + L['dt']) * L['a']
        C = qkv.shape[-1]
        w = L['conv'].reshape(C, 1, self.d_conv).to(qkv.dtype)
        c = F.conv1d(F.pad(qkv.t()[None], (self.d_conv - 1, 0)), w, groups=C)[0].t()
        c = F.silu(c)
        nkd = self.hk * self.nk
        q, k, v = c[:, :nkd].view(T, self.nk, self.hk), c[:, nkd:2 * nkd].view(T, self.nk, self.hk), c[:, 2 * nkd:].view(T, self.nv, self.hv)
        q = (q.float() / q.float().norm(dim=-1, keepdim=True).clamp(min=self.eps)).to(x.dtype)
        k = (k.float() / k.float().norm(dim=-1, keepdim=True).clamp(min=self.eps)).to(x.dtype)
        if self.nv != self.nk:                   # tiled: v head j uses k head j % nk (ggml_repeat / fused kernel)
            q, k = q.repeat(1, self.nv // self.nk, 1), k.repeat(1, self.nv // self.nk, 1)
        o = gdn(q[None].contiguous(), k[None].contiguous(), v[None].contiguous(), g[None], beta[None], self.hk ** -0.5)[0]
        o = rms(o, L['ssm_norm'], self.eps) * F.silu(z.view(T, self.nv, self.hv))
        return L['ssm_out'](o.reshape(T, self.nv * self.hv))

    def block(self, i, h):
        L = self.layers[i]
        x = rms(h, L['attn_norm'], self.eps)
        h = h + (self.gdn_layer(L, x) if L['recr'] else self.attn(L, x))
        x = rms(h, L['post_norm'], self.eps)
        xr = self.shared_rot(x, [L['ffn_gate'], L['ffn_up']])
        return h + L['ffn_down'](F.silu(L['ffn_gate'](x, xr)) * L['ffn_up'](x, xr))

    def hidden(self, ids, ckpt=False):
        h = self.E[ids]
        if self.embed_rot is not None:
            h = self.embed_rot(h)
        for i in range(self.n_layer):
            h = checkpoint(self.block, i, h, use_reentrant=False) if ckpt else self.block(i, h)
        return rms(h, self.out_norm, self.eps)

    def logits(self, h):
        if self.head is not None:
            return self.head(h).float()
        x = self.head_rot(h) if self.head_rot is not None else h
        return (x @ self.E.t()).float()

    def loss(self, ids, labels, ckpt=True, chunk=1024):
        """Mean cross-entropy over positions with labels != -100 (labels already shifted: labels[t] = ids[t+1])."""
        h = self.hidden(ids, ckpt)
        n = int((labels != -100).sum())
        tot = h.new_zeros((), dtype=torch.float32)
        for s in range(0, h.shape[0], chunk):
            lab = labels[s:s + chunk]
            if bool((lab != -100).any()):
                f = lambda hh, ll: F.cross_entropy(self.logits(hh), ll, ignore_index=-100, reduction='sum')
                tot = tot + (checkpoint(f, h[s:s + chunk], lab, use_reentrant=False) if ckpt else f(h[s:s + chunk], lab))
        return tot / max(n, 1), n

    @torch.no_grad()
    def greedy(self, ids, n_new, stop=()):
        ids = list(ids)
        for _ in range(n_new):
            t = torch.tensor(ids, device=self.device)
            nxt = int(self.logits(self.hidden(t)[-1:]).argmax(-1))
            ids.append(nxt)
            if nxt in stop:
                break
        return ids

    # -- bookkeeping
    def set_voting(self, on):
        for m in self.tlayers:
            m.voting = on

    def state_bytes(self):
        return sum(m.counter.numel() + m.packed.numel() for m in self.tlayers)

    def flips(self):
        return sum(m.flips for m in self.tlayers)

    def save_codes(self, path):
        torch.save({m.name: m.packed.cpu() for m in self.tlayers}, path + '.tmp')
        os.replace(path + '.tmp', path)

    def load_codes(self, path):
        d = torch.load(path)
        for m in self.tlayers:
            m.packed.copy_(d[m.name].to(m.packed.device))

    def export_gguf(self, out_path):
        """Copy the GGUF and overwrite the code bytes of every trainable tensor (scales unchanged)."""
        from gguf import GGUFReader
        shutil.copyfile(self.path, out_path)
        r = GGUFReader(self.path)
        T = {t.name: t for t in r.tensors}
        mm = np.memmap(out_path, dtype=np.uint8, mode='r+')
        changed = total = 0
        for m in self.tlayers:
            t = T[m.name]
            blocks = mm[t.data_offset:t.data_offset + t.n_bytes].reshape(m.out_f, m.in_f // QK, 2 + QK // 4)
            new = m.packed.cpu().numpy().reshape(m.out_f, m.in_f // QK, QK // 4)
            old = torch.from_numpy(np.ascontiguousarray(blocks[..., 2:]).reshape(m.out_f, -1))
            changed += int((unpack(old) != unpack(m.packed.cpu())).sum()); total += m.out_f * m.in_f
            blocks[..., 2:] = new
        mm.flush(); del mm
        r2 = GGUFReader(out_path)                 # read back and compare
        T2 = {t.name: t for t in r2.tensors}
        for m in self.tlayers[:3] + self.tlayers[-3:]:
            pk, _ = pq2_split(np.array(T2[m.name].data), m.out_f, m.in_f)
            assert np.array_equal(pk, m.packed.cpu().numpy()), f'export mismatch in {m.name}'
        return changed, total


# ----------------------------------------------------------------------------- data
ASSIST = re.compile(r'<\|im_start\|>assistant\n(.*?<\|im_end\|>)', re.S)


def norm_messages(msgs):
    out = []
    for m in msgs:
        m = {k: v for k, v in m.items() if v is not None}
        m.setdefault('content', '')
        if m.get('tool_calls'):
            tcs = []
            for tc in m['tool_calls']:
                tc = dict(tc)
                fn = dict(tc.get('function', tc))
                a = fn.get('arguments')
                if isinstance(a, str):
                    try:
                        fn['arguments'] = json.loads(a)
                    except Exception:
                        pass
                tcs.append({'type': 'function', 'function': {'name': fn.get('name'), 'arguments': fn.get('arguments', {})}})
            m['tool_calls'] = tcs
        out.append(m)
    return out


def encode(tok, ex, max_len):
    """Chat-template the trajectory; train only on assistant turns. Returns (ids, labels) or None."""
    try:
        text = tok.apply_chat_template(norm_messages(ex['messages']), tools=ex.get('tools') or None, tokenize=False)
    except Exception:
        return None
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc['input_ids'][:max_len + 1], enc['offset_mapping'][:max_len + 1]
    spans = [(m.start(1), m.end(1)) for m in ASSIST.finditer(text)]
    tgt = np.zeros(len(ids), dtype=bool)
    j = 0
    for i, (a, _) in enumerate(offs):
        while j < len(spans) and spans[j][1] <= a:
            j += 1
        if j < len(spans) and spans[j][0] <= a < spans[j][1]:
            tgt[i] = True
    labels = [ids[i + 1] if tgt[i + 1] else -100 for i in range(len(ids) - 1)]
    if sum(l != -100 for l in labels) < 8:
        return None
    return ids[:-1], labels


def build_data(tok, configs, n_train, n_eval, max_len, seed=0):
    from datasets import load_dataset
    train, ev, stats = [], [], {}
    for cfg in configs:
        ds = load_dataset('openbmb/UltraData-SFT-Agent-2609', cfg, split='train', streaming=True)
        ds = ds.shuffle(seed=seed, buffer_size=5000)
        want_tr, want_ev = n_train // len(configs), n_eval // len(configs)
        a, b, skipped = [], [], 0
        for ex in ds:
            e = encode(tok, ex, max_len)
            if e is None:
                skipped += 1; continue
            (b if len(b) < want_ev else a).append(e)
            if len(a) >= want_tr and len(b) >= want_ev:
                break
        stats[cfg] = dict(train=len(a), eval=len(b), skipped=skipped,
                          mean_tokens=float(np.mean([len(x[0]) for x in a])), mean_targets=float(np.mean([sum(l != -100 for l in x[1]) for x in a])))
        train += a; ev += b
    rng = np.random.default_rng(seed)
    rng.shuffle(train)
    return train, ev, stats


@torch.no_grad()
def eval_loss(model, data, device):
    tot = n = 0
    for ids, lab in data:
        l, k = model.loss(torch.tensor(ids, device=device), torch.tensor(lab, device=device), ckpt=False)
        tot += float(l) * k; n += k
    return tot / max(n, 1)


@torch.no_grad()
def ppl_llamacpp_style(model, ids, n_ctx=512, n_chunks=16):
    """llama-perplexity: independent chunks of n_ctx tokens, score the second half of each."""
    nll = cnt = 0
    first = n_ctx // 2
    for c in range(min(n_chunks, len(ids) // n_ctx)):
        t = torch.tensor(ids[c * n_ctx:(c + 1) * n_ctx], device=model.device)
        lg = model.logits(model.hidden(t))
        lp = torch.log_softmax(lg[first:n_ctx - 1], -1)
        tgt = t[first + 1:n_ctx]
        nll -= float(lp.gather(1, tgt[:, None]).sum()); cnt += len(tgt)
    return math.exp(nll / cnt)
