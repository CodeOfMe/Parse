# Needle: Simple Attention Networks for Function Calling

> **GitHub:** [cactus-compute/needle](https://github.com/cactus-compute/needle)
> **HuggingFace:** [Cactus-Compute/needle](https://huggingface.co/Cactus-Compute/needle)
> **Architecture Docs:** [docs/simple_attention_networks.md](https://github.com/cactus-compute/needle/blob/main/docs/simple_attention_networks.md)
> **PDF:** N/A (GitHub project, no standalone paper)

---

## Abstract

Needle is a distilled 26M parameter "Simple Attention Network" based on Gemini 3.1, designed for function calling tasks. In production, Needle runs on Cactus at 6000 toks/sec prefill and 1200 decode speed. The core insight is that MLPs can be completely dropped from transformer networks when the model relies on external knowledge sources. Since function calling relies on external tool lists, Needle was designed as a simple attention network specifically for function calling, distilled from Gemini-3.1-Flash-Lite.

## Architecture

### Simple Attention Network Design

The architecture is an encoder-decoder transformer with shared embeddings, tied output, RoPE, and bfloat16 precision:
- **Encoder**: 12 layers, GQA (8H/4KV), RoPE, gated residuals, **no FFN**
- **Decoder**: 8 layers, self-attention + cross-attention, gated residuals, **no FFN**
- **d_model**: 512
- **Vocab**: 8192 (SentencePiece BPE)
- **Norm**: ZCRMSNorm (zero-centered, init=0)
- **Precision**: bfloat16 (INT4 QAT during training)

### Why No FFN

1. **Softmax is nonlinear**: `softmax(QK^T/sqrt(d)) * V` is a data-dependent nonlinear mixing operation. For tasks about routing information (query → tool alignment), attention is the right primitive.
2. **Tool calling is retrieval-and-assembly**: Matching query to tool name, extracting argument values, and assembling JSON are all aligning and copying between input and output — exactly what cross-attention does. No step requires per-position feature transformation (which is what FFN provides).
3. **At small scale, FFN parameters are wasted**: ~2/3 of standard transformer parameters are FFN. For a <50M model on a structured task, those parameters contribute less than more attention layers (deeper cross-attention = better query-tool alignment).
4. **Fewer parameters = faster inference**: FFNs have the biggest GEMM/GEMV dimensions — removing them cuts per-layer parameters by ~2/3, directly reducing the memory bandwidth bottleneck that dominates latency on edge devices.

### Why Encoder-Decoder

1. **Bidirectional encoding**: Tools are structured objects — a bidirectional encoder sees the full definition at once. A causal model sees it left-to-right and must infer structure from partial context.
2. **No input tokens in KV cache**: Encoder-decoder uses a fixed-size encoder representation for cross-attention, not re-attending the full input at every generation step.
3. **Natural fit for multi-head design**: The encoder output feeds the decoder (generation) and the contrastive head (tool retrieval). Clean separation.

### Gated Residuals

Without FFN, there is no per-position nonlinear rewriting per layer. This makes residual connection design critical:
- **Standard residual**: `x = x + Attn(Norm(x))` — attention can only ADD a delta. Without FFN to do the rewriting, purely additive is limiting.
- **No residual**: `x = Attn(Norm(x))` — each layer fully rewrites, but we lose the gradient highway. Deep networks (12+ layers) will not train.
- **Gated residual (ours)**: `x = x + sigmoid(gate) * Attn(Norm(x))` — per-sublayer learnable scalar, initialized to 0. sigmoid(0) = 0.5, so training starts with half-strength residual. The model can learn to sharpen useful layers (g→1) or suppress unhelpful ones (g→0) without losing gradient flow.

### ZCRMSNorm

- **Standard RMSNorm**: `x * gamma / RMS(x)`, gamma initialized to 1.
- **ZCRMSNorm**: `x * (1 + gamma) / RMS(x)`, gamma initialized to 0.
- At init, ZCRMSNorm is identity-up-to-scale. Pairs with gated residuals: the entire block starts as a damped identity + damped normalized attention. No component starts with a strong learned bias.
- From the nGPT / DeepSeek-V3 line of work. Applied to QK heads as well (QK-norm) for training stability.

### Contrastive Tool Selection Head

CLIP-style head for retrieving relevant tools before generation. Useful when the tool set is large and you want to filter to the top-k most relevant tools for a query.
- **Architecture**: encoder output → mean pool over non-pad positions → Dense(d_model/4) → ReLU → Dense(128) → L2-normalize. Produces a unit vector per input.
- **Training**: symmetric contrastive loss (CLIP). Each batch pairs queries with their positive tools; in-batch negatives provide the contrastive signal. Learnable temperature (`log_temp`).
- **Inference**: encode query and each tool candidate into the shared embedding space, rank by cosine similarity, take top-k.
- Trained jointly with the main CE loss at 0.1x weight. Same encoder is used for both generation and retrieval.

### Muon for Attention-Only

- **Dual optimizer**: Muon (Q/K/V/O projections, LR 0.02, WD 0.01) + AdamW (everything else, LR 3e-4).
- Without FFN, the model is a deep stack of linear projections with softmax routing. Muon enforces orthogonality on weight updates via Newton-Schulz, preventing the representation collapse that can happen when stacking many linear layers without interleaving nonlinearities.

### INT4 QAT as Regularization

- **Fake quantization every 100 steps**: Weights are group-wise quantized to INT4 (symmetric, group_size=32) with straight-through estimator (STE), then used for the forward pass. Gradients flow through the rounding via STE.
- **Regularization effect**: Quantization noise acts as a form of weight noise regularization, discouraging the model from relying on precise weight values. For a small model with no FFN (fewer parameters, higher overfitting risk), this is beneficial.
- **Deploy-ready**: The model trains with the same quantization it will see at inference. No post-training quantization gap.

### Token-Level Loss Weighting

- Base (JSON structure): 1.0x, tool names: 2.0x, argument keys: 1.5x, argument values: 4.0x.
- The model achieves ~99% JSON parse rate early in training. The actual errors are values > names > keys > structure. Weighting matches the error distribution.
- Auxiliary: z-loss for logit stability, CLIP contrastive loss at 0.1x.

## Training

- Pretrained on 16 TPU v6e for 200B tokens (27hrs).
- Post-trained on 2B tokens of single-shot function call dataset (45mins).

## Performance

Needle (26M) beats FunctionGemma-270m, Qwen-0.6B, Granite-350m, LFM2.5-350m on single-shot function call for personal AI, while being an order of magnitude smaller.

## Citation

```bibtex
@misc{ndubuaku2026needle,
    title  = {Needle: Simple Attention Networks for Function Calling},
    author = {Henry Ndubuaku and Jakub Mroz and Karen Mosoyan and Roman Shemet and Parkirat Sandhu and Satyajit Kumar and Noah Cylich and Justin H. Lee},
    year   = {2026},
    url    = {https://github.com/cactus-compute/needle},
    note   = {GitHub repository, accessed 2026}
}
```
