# Enhancing Robot Instruction Understanding and Disambiguation via Speech Prosody

This repository contains the code and dataset accompanying our INTERSPEECH 2025 paper:

**Title:** Enhancing Robot Instruction Understanding and Disambiguation via Speech Prosody  
**Authors:** David Sasu, Kweku Andoh Yamoah, Benedict Quartey, Natalie Schluter  
**Affiliations:** IT University of Copenhagen, University of Florida, Brown University  

## 📄 Abstract

Enabling robots to accurately interpret and execute spoken language instructions is vital for human-robot collaboration. Traditional approaches discard speech prosody during transcription, leading to ambiguity in understanding user intent. We present a novel approach that **leverages speech prosody to detect referent intents** and integrates them with **large language models (LLMs)** to disambiguate and select the correct task plan.

Key contributions include:
- A **prosody-aware encoder-decoder model** (BiLSTM and Transformer-based) for per-word intent labeling.
- An **LLM integration module** using in-context learning for robot plan selection.
- A **new dataset of 1,540 ambiguous spoken instructions** annotated for prosodic referents.

Our approach achieves **95.79% token-level referent intent accuracy** and **71.96% task plan disambiguation accuracy** using GPT-4o and prosody-based predictions.

## 📦 Contents

- `models/` – Prosody-aware BiLSTM and Transformer architectures for referent detection.
- `data/` – Speech dataset of ambiguous instructions (1,540 samples).
- `scripts/` – Training, evaluation, and LLM integration scripts.
- `notebooks/` – Exploratory notebooks for dataset analysis and ablation studies.

## 🧠 Method Overview

### 1. Referent Detection

We fuse **prosodic features** (extracted via [Disvoice](https://disvoice.readthedocs.io/en/latest/index.html)) with **text embeddings** (OpenAI Text Embedding 3) and process them through either:

- **BiLSTM Encoder-Decoder** with attention fusion
- **Transformer Encoder-Decoder** with positional encoding and self-attention

Each word in an instruction is classified as either:
- **Goal referent** (target object/location)
- **Detail referent** (qualifiers)
- **Non-referent**

### 2. Task Plan Disambiguation via LLM

Detected intents are used to construct prompts for LLMs (e.g., GPT-4o, o1-mini) to choose the correct robot task plan from multiple candidates.

## 🗃️ Dataset

- **Size:** 1,540 utterances (121 minutes)
- **Source:** 22 speakers (age 18–22) each recorded 35 ambiguous instructions with both interpretations
- **Format:** Audio files, aligned transcripts, prosody annotations, referent labels

We present this dataset as the **first speech disambiguation corpus for robotics**.

## 📊 Results

### Referent Intent Classification (Best Model: BiLSTM)
- **Goal F1:** 79.15%
- **Detail F1:** 83.24%
- **Overall Accuracy:** 95.79%

### Task Plan Disambiguation (LLM + Prosody + Transformer)
- **GPT-4o:** 71.96%
- **o1-mini:** 62.5%
- **o3-mini:** 61.83%

## 📢 Citation
If you use this work, please cite:
```bibtex
@inproceedings{sasu2025prosody,
  title={Enhancing Robot Instruction Understanding and Disambiguation via Speech Prosody},
  author={Sasu, David and Yamoah, Kweku Andoh and Quartey, Benedict and Schluter, Natalie},
  booktitle={Proceedings of Interspeech 2025},
  year={2025}
}
```


## 📬 Contact
- David Sasu (dasa@itu.dk)
- Kweku Andoh Yamoah (kyamoah@ufl.edu)
- Benedict Quartey (benedictquartey@brown.edu)
- Natalie Schluter (natschluter@apple.com)
