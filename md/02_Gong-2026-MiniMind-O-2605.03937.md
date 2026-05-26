# MiniMind-O Technical Report: An Open Small-Scale Speech-Native Omni Model

> **arXiv:** [2605.03937](https://arxiv.org/abs/2605.03937)
> **GitHub:** [jingyaogong/minimind-o](https://github.com/jingyaogong/minimind-o)
> **PDF:** [Gong-2026-MiniMind-O-2605.03937.pdf](56_Gong-2026-MiniMind-O-2605.03937.pdf)

---

## Abstract

This open-source project aims to fully implement a small-scale end-to-end Omni model from scratch, supporting text/audio/image tri-modal input and text/streaming speech output with a single weight. The `minimind-3o` model is only ~0.1B parameters, allowing training on consumer GPUs (single RTX 3090) and fast CPU inference. It represents the smallest complete Omni implementation publicly available. The project provides both mini and full training datasets, complete code and technical reports covering the Thinker–Talker dual-path architecture, streaming speech generation, real-time barge-in, near-duplex interaction, voice cloning, and phone-mode WebUI. All core algorithms are implemented from scratch using native PyTorch without third-party high-level abstractions.

## Introduction

Following MiniMind (LLM) and MiniMind-V (VLM), MiniMind-O is the third installment in the series. "Omni" here means enabling a single model to simultaneously possess multi-modal interaction capabilities of listening, seeing, and speaking: receiving text, speech, and visual signals, and outputting text and streaming speech. While GPT-4o first brought natural streaming speech interaction to attention, followed by open-source works like Mini-Omni2, Moshi, GLM-4-Voice, and Qwen3-Omni, the community still lacks a lightweight, complete starting point for understanding, training, and modifying an Omni model from scratch. MiniMind-O attempts to fill this gap by connecting speech and text directly at the hidden state level, maintaining an end-to-end Omni pipeline with only a 0.1B backbone.

## Architecture

### Thinker–Talker Dual-Path

The main body consists of two paths:
- **Thinker**: Responsible for understanding text, speech, and image inputs, and generating semantic text responses. Text tokens enter the language backbone directly; speech and image features are extracted by Audio/Vision Encoders and mapped to the MiniMind hidden space via projectors.
- **Talker**: Generates 8-layer Mimi audio codes from the semantic state provided by Thinker using Multi-Token Prediction (MTP). This is not a simple cascade of ASR → LLM → TTS, but a unified sequence preserving text reasoning, speech output, and streaming interaction capabilities.

### Bridge Layer

The representation passed from Thinker to Talker is taken from a middle hidden layer (`bridge_layer = num_hidden_layers // 2 - 1`), not the embedding or final layer. The embedding layer lacks sufficient semantic information, while the final layer is too close to the next-token prediction objective. The middle layer has already fused context and cross-modal information without being over-shaped by the LM head, making it more suitable as a condition for speech generation.

### Talker Speech Generation

The Talker converts the semantic state from Thinker into an 8-layer Mimi codebook sequence. MTP is used to simultaneously predict multiple audio codebooks rather than decomposing each codebook into an independent long chain. To control additional parameters in the 0.1B model, audio embeddings and output heads adopt a shared main body with lightweight codebook adapters, preserving distribution differences across codebooks while avoiding duplicating a full set of parameters for each codebook layer.

### Sequence Format and Streaming Decoding

Text tokens and 8 audio-code streams are placed in the same training sample: Thinker handles the text sequence, Talker handles the audio code sequence, and speech, image, and voice conditions are injected via placeholders or reference codes. Loss is computed only after the reply begins, so reference and conditioning regions only provide conditions, not reconstruction targets. During streaming generation, the model produces text tokens while补齐 8-layer Mimi codes via MTP and delayed scheduling. The Mimi decoder can incrementally restore 24 kHz waveforms, so speech playback does not need to wait for the complete answer.

### Voice Control

Voice control is achieved through in-context voice cloning: reference audio is encoded into a voice prompt and fed to Talker as a contextual condition, rather than fine-tuning weights or rewriting text prompts. The model can also use speaker embeddings for more stable speaker constraints. Changing voice at inference only requires replacing these condition information; Thinker prompts and Talker weights remain unchanged.

## Training Pipeline

The full training does not decompose into complex multi-stage pre-training, but progressively接入 capabilities according to data flow:
1. **sft_t2a**: Align text-to-speech output first, letting Talker learn to generate Mimi codes under Thinker semantic conditions.
2. **sft_a2a**: Then接入 speech input, enabling the model to enter the same Thinker–Talker reply chain from speech instructions.
3. **sft_i2t**: Finally align the vision path, where `vision_proj` mode only updates the vision projection layer to avoid image data from overwriting language and speech capabilities.

Audio Encoder, Vision Encoder, and Speech Codec remain frozen throughout. Dense and MoE versions follow the same data order.

## Experiments

### Talker Hidden Size Ablation

If only speech generation is considered, Talker at 1024/2048 dimensions or adding more layers would be more stable. But MiniMind-O aims to compress the complete Omni pipeline to ~0.1B, so most parameters cannot be given to the acoustic side. After Thinker/Talker decoupling, language understanding and cross-modal fusion are mainly handled by Thinker, while Talker only renders Mimi codes on semantic conditions, making a small Talker possible. 384 dimensions are most attractive (dense version can be compressed to ~88M), but results show that small does not automatically mean cost-effective: short sentences can be maintained, but medium/long sentences are more prone to word omission, repetition, and pronunciation drift. 768 dimensions were retained because they match the MiniMind backbone dimension, allowing initialization with the last 4 layers of Thinker; parameters remain around 0.1B, training cost does not increase significantly, and consistency is much more stable.

| Variant | Talker hidden | Params | Avg CER ↓ | Short ↓ | Mid / Long ↓ |
|---|---|---|---|---|---|
| Dense | 768 | 115.29M | **0.0897** | 0.1528 | 0.0874 / 0.0675 |
| Dense | 512 | 96.13M | 0.1745 | 0.2709 | 0.2455 / 0.0976 |
| Dense | 384 | 88.72M | 0.2767 | 0.3904 | 0.1865 / 0.4046 |
| MoE | 768 | 317.05M-A115.33M | **0.0900** | 0.2075 | 0.0533 / 0.0271 |
| MoE | 512 | 261.32M-A96.17M | 0.1265 | 0.0711 | 0.1490 / 0.1464 |
| MoE | 384 | 240.04M-A88.75M | 0.3280 | 0.3757 | 0.2777 / 0.4313 |

### Voice Cloning Similarity

Voice cloning is a beta capability in the current version. Most open-source Omni models only support fixed output voices, while minimind-3o attempts to fit multi-voice generation into the same Talker. This goal is harder than "being able to speak" because the model must not only say the content correctly but also preserve speaker voice clues when generating Mimi codes. Current effects are not yet high-fidelity cloning; the same reference voice does not always remain consistent across different questions, and long sentences are easily affected by pronunciation and rhythm issues. But basic male/female voice differences, intonation tendencies, and some prosodic features can be distinguished. CAM++ speaker embedding cosine similarity is provided as an automated reference.

### Cross-Model English T2A Comparison

20 English questions were selected with the constraint `Answer briefly in one short sentence` to keep response lengths in the same range. After three models generated audio, they were uniformly transcribed by Qwen3-ASR and CER/WER was calculated against target text to compare Talker text consistency.

| Length bucket | Mini-Omni CER/WER | Mini-Omni2 CER/WER | minimind-3o CER/WER |
|---|---|---|---|
| short (≤15w) | 0.0195 / 0.0384 (n=8) | 0.0503 / 0.0584 (n=14) | 0.0531 / 0.0417 (n=8) |
| mid (16–30w) | 0.0038 / 0.0052 (n=12) | 0.0062 / 0.0076 (n=6) | 0.1327 / 0.1420 (n=11) |
| long (31–60w) | — | — | 0.0431 / 0.0508 (n=1) |

In short replies ≤15 words, minimind-3o already approaches Mini-Omni2. The real gap opens in the 16–30 word segment, where Talker needs to maintain pronunciation, rhythm, and word consistency in a complete sentence, which is also the interval where the current 0.1B Talker is most prone to instability.

## Conclusion and Future Work

The current model still has gaps compared to large-scale Omni systems in long speech naturalness, complex visual reasoning, open-ended English long answers, and voice stability. These limitations also point to future directions: longer ICL context, finer prosody supervision, stronger vision encoders, more stable voice conditions, and systematic scanning of Bridge layer and MTP codebook interface.

The value of MiniMind-O lies precisely here: it compresses a complete Omni closed-loop to the 0.1B scale and places code, weights, and main training data in a single inspectable object. This means it is not just a demo, but a baseline small enough and transparent enough to be truly reproduced and further modified from scratch. For those who want to understand details like Thinker–Talker decoupling, MTP codebook interface, in-context voice cloning, and middle hidden bridge, it provides a set of design experiences that can be truly verified hands-on.

---

## Citation

```bibtex
@article{minimind-o-report,
    title   = {MiniMind-O Technical Report: An Open Small-Scale Speech-Native Omni Model},
    author  = {Jingyao Gong},
    journal = {arXiv preprint arXiv:2605.03937},
    year    = {2026}
}

@misc{minimind-o,
    title  = {MiniMind-O: Train a Tiny Omni Model from Scratch},
    author = {Jingyao Gong},
    year   = {2026},
    url    = {https://github.com/jingyaogong/minimind-o},
    note   = {GitHub repository, accessed 2026}
}
```
