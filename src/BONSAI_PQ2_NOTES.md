# Bonsai 2 9B counter fine-tune: findings (paused 2026-10-23, for a later paper section)

- Weights: prism-ml/Ternary-Bonsai-2-9B-gguf, file PQ2_0.gguf (2.42 GB). No unpacked safetensors exist.
- PQ2_0 block (PrismML llama.cpp fork, branch `prism`, ggml-common.h): fp16 scale d + 32 bytes, 128 weights,
  2 bits each, element j -> byte j//4, shift (j%4)*2; value = (q - 1) * d (00=-1, 01=0, 10=+1; 11=+2 unused).
  gguf-py in the fork knows type 142 (PQ2_0).
- Hadamard folding (src/llama-model.cpp, llama-graph.cpp build_lora_mm): for weights in prism.hadamard.weight_names,
  y = W_stored @ H_b(signs * perm(x)); H_b = block-diagonal normalized Sylvester Hadamard (entry (-1)^popcount(r&c)/sqrt(b)),
  b = prism.hadamard.block_size; signs per input width (sign_mode explicit: sign_widths/sign_values).
  perm only for ssm_out when prism.hadamard.gdn_v_grouped: tiled [hd, nk, rep] -> grouped [hd, rep, nk].
  Embedding in inverse_weight_names: h = signs * (H z) after lookup. Version 2 = tied output: logits = E_stored @ H(signs * h).
- Architecture (src/models/qwen35.cpp): 32 layers, 3 GDN : 1 full attention; attn: wq gives [q, gate] per head
  (stride 2*head_dim), q/k RMS norm, IMROPE (text: plain NEOX rope on n_rot dims), out * sigmoid(gate);
  GDN: qkv -> causal depthwise conv (left pad K-1) -> silu -> l2norm q,k (x / max(|x|, eps)); q,k tiled to v heads
  (v head j uses k head j % n_k); g = softplus(alpha + dt_bias) * ssm_a (ssm_a = -exp(A_log)); beta = sigmoid;
  delta rule with q scale 1/sqrt(d_k); gated RMS norm * silu(z); ssm_out. FFN SwiGLU. Norm weights used as stored.
- Plan: own PyTorch port + flash-linear-attention chunk_gated_delta_rule; counter updated inside each layer's
  backward (no full gradient buffer); packed 2-bit codes; embedding frozen; check vs fork's llama-perplexity and
  llama-simple before training; write trained codes back into the GGUF (same block layout) for CPU release.
- Data: openbmb/UltraData-SFT-Agent-2609 (Tool-Use, Code-Agent; streaming). LICENSE WARNING: card says public access
  "is not a grant of redistribution, commercial use, or training rights" -> get written permission before releasing a model.
