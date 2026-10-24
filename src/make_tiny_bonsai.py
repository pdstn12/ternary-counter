"""Write a tiny random Ternary-Bonsai-2-style GGUF (arch qwen35, PQ2_0 weights, Hadamard folding with explicit signs,
GDN v-head grouping, tied output, version-2 contract). Used to check the PyTorch port against PrismML's llama.cpp.
Usage: python make_tiny_bonsai.py OUT.gguf   (needs the fork's gguf-py on the path)"""
import sys
import numpy as np
import gguf
from gguf import GGUFWriter, GGMLQuantizationType as Q

out = sys.argv[1]
rng = np.random.default_rng(1234)
E, NL, NH, NKV, HD, NROT = 256, 4, 2, 1, 128, 32
HK, NK, NV, KC, FF, B = 64, 2, 4, 4, 512, 64
INNER = HK * NV
SECTIONS = [6, 5, 5, 0]
QK = 128


def pq2(rows, cols, scale):
    """random ternary codes + fp16 group scales, laid out as PQ2_0 blocks: bytes [rows, cols/128 * 34]"""
    codes = rng.choice([-1, 0, 1], size=(rows, cols), p=[0.3, 0.4, 0.3])
    v = (codes + 1).astype(np.uint8).reshape(rows, cols // QK, QK // 4, 4)
    qs = v[..., 0] | (v[..., 1] << 2) | (v[..., 2] << 4) | (v[..., 3] << 6)
    d = (scale * (0.5 + rng.random((rows, cols // QK)))).astype(np.float16)
    blocks = np.concatenate([d.view(np.uint8).reshape(rows, cols // QK, 2), qs], -1)
    return blocks.reshape(rows, -1)


w = GGUFWriter(out, 'qwen35')
w.add_string('general.name', 'tiny-bonsai-test')
A = 'qwen35.'
for k, v in dict(context_length=4096, embedding_length=E, block_count=NL, feed_forward_length=FF,
                 full_attention_interval=4).items():
    w.add_uint32(A + k, v)
w.add_uint32(A + 'attention.head_count', NH); w.add_uint32(A + 'attention.head_count_kv', NKV)
w.add_uint32(A + 'attention.key_length', HD); w.add_uint32(A + 'attention.value_length', HD)
w.add_float32(A + 'attention.layer_norm_rms_epsilon', 1e-6)
w.add_uint32(A + 'rope.dimension_count', NROT); w.add_float32(A + 'rope.freq_base', 10000.0)
w.add_array(A + 'rope.dimension_sections', SECTIONS)
w.add_uint32(A + 'ssm.conv_kernel', KC); w.add_uint32(A + 'ssm.inner_size', INNER)
w.add_uint32(A + 'ssm.state_size', HK); w.add_uint32(A + 'ssm.time_step_rank', NV); w.add_uint32(A + 'ssm.group_count', NK)
# byte-level vocab: 256 GPT-2 byte symbols + one special
bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
cs, n = bs[:], 0
for b in range(256):
    if b not in bs:
        bs.append(b); cs.append(256 + n); n += 1
byte2uni = dict(zip(bs, [chr(c) for c in cs]))
toks = [byte2uni[b] for b in range(256)] + ['<|endoftext|>']
w.add_string('tokenizer.ggml.model', 'gpt2'); w.add_string('tokenizer.ggml.pre', 'default')
w.add_array('tokenizer.ggml.tokens', toks); w.add_array('tokenizer.ggml.token_type', [1] * 256 + [3])
w.add_array('tokenizer.ggml.merges', [byte2uni[ord('t')] + ' ' + byte2uni[ord('h')]][:0] or ['a b'])
w.add_uint32('tokenizer.ggml.bos_token_id', 256); w.add_uint32('tokenizer.ggml.eos_token_id', 256)
w.add_bool('tokenizer.ggml.add_bos_token', False)
V = len(toks)

names, tensors = [], []


def add(name, arr, raw=None):
    tensors.append((name, arr, raw))


def fold(name):
    names.append(name)


add('token_embd.weight', pq2(V, E, 0.15), Q.PQ2_0)
add('output_norm.weight', (1 + 0.1 * rng.standard_normal(E)).astype(np.float32))
for i in range(NL):
    p = f'blk.{i}.'
    add(p + 'attn_norm.weight', (1 + 0.1 * rng.standard_normal(E)).astype(np.float32))
    add(p + 'post_attention_norm.weight', (1 + 0.1 * rng.standard_normal(E)).astype(np.float32))
    for nm, (r, c) in dict(ffn_gate=(FF, E), ffn_up=(FF, E), ffn_down=(E, FF)).items():
        add(p + nm + '.weight', pq2(r, c, 1.5 / np.sqrt(c)), Q.PQ2_0); fold(p + nm + '.weight')
    if (i + 1) % 4 != 0:                                   # gated delta net layer
        add(p + 'attn_qkv.weight', pq2(2 * HK * NK + INNER, E, 1.5 / np.sqrt(E)), Q.PQ2_0); fold(p + 'attn_qkv.weight')
        add(p + 'attn_gate.weight', pq2(INNER, E, 1.5 / np.sqrt(E)), Q.PQ2_0); fold(p + 'attn_gate.weight')
        add(p + 'ssm_conv1d.weight', (0.4 * rng.standard_normal((2 * HK * NK + INNER, KC))).astype(np.float32))
        add(p + 'ssm_dt.bias', rng.standard_normal(NV).astype(np.float32))
        add(p + 'ssm_a', (-np.exp(0.5 * rng.standard_normal(NV))).astype(np.float32))
        add(p + 'ssm_beta.weight', (0.1 * rng.standard_normal((NV, E))).astype(np.float32))
        add(p + 'ssm_alpha.weight', (0.1 * rng.standard_normal((NV, E))).astype(np.float32))
        add(p + 'ssm_norm.weight', (1 + 0.1 * rng.standard_normal(HK)).astype(np.float32))
        add(p + 'ssm_out.weight', pq2(E, INNER, 1.5 / np.sqrt(INNER)), Q.PQ2_0); fold(p + 'ssm_out.weight')
    else:                                                  # full attention layer
        add(p + 'attn_q.weight', pq2(NH * HD * 2, E, 1.5 / np.sqrt(E)), Q.PQ2_0); fold(p + 'attn_q.weight')
        add(p + 'attn_k.weight', pq2(NKV * HD, E, 1.5 / np.sqrt(E)), Q.PQ2_0); fold(p + 'attn_k.weight')
        add(p + 'attn_v.weight', pq2(NKV * HD, E, 1.5 / np.sqrt(E)), Q.PQ2_0); fold(p + 'attn_v.weight')
        add(p + 'attn_output.weight', pq2(E, NH * HD, 1.5 / np.sqrt(NH * HD)), Q.PQ2_0); fold(p + 'attn_output.weight')
        add(p + 'attn_q_norm.weight', (1 + 0.1 * rng.standard_normal(HD)).astype(np.float32))
        add(p + 'attn_k_norm.weight', (1 + 0.1 * rng.standard_normal(HD)).astype(np.float32))

widths = sorted({E, FF, INNER, NH * HD})
w.add_uint32('prism.hadamard.version', 2); w.add_bool('prism.hadamard.tied_output', True)
w.add_uint32('prism.hadamard.block_size', B)
w.add_string('prism.hadamard.transform', 'normalized-sylvester-walsh-hadamard')
w.add_string('prism.hadamard.axis', 'input-last-dimension'); w.add_string('prism.hadamard.sign_mode', 'explicit')
w.add_array('prism.hadamard.weight_names', names)
w.add_array('prism.hadamard.inverse_weight_names', ['token_embd.weight'])
w.add_array('prism.hadamard.sign_widths', widths)
w.add_array('prism.hadamard.sign_values', [int(x) for wd in widths for x in rng.choice([-1, 1], wd)])
w.add_bool('prism.hadamard.gdn_v_grouped', True)
F32 = '--f32' in sys.argv            # same values as float32 tensors: llama.cpp then runs without activation quantization
for name, arr, raw in tensors:
    if F32 and raw == Q.PQ2_0:
        rows = arr.shape[0]
        bl = arr.reshape(rows, -1, 34)
        d = np.ascontiguousarray(bl[..., :2]).view(np.float16).astype(np.float32)
        q = np.stack([(bl[..., 2:] >> s) & 3 for s in (0, 2, 4, 6)], -1).reshape(rows, -1, QK).astype(np.float32) - 1
        w.add_tensor(name, (q * d).reshape(rows, -1))
    else:
        w.add_tensor(name, arr, raw_dtype=raw)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print('wrote', out, len(tensors), 'tensors,', len(names), 'folded')
