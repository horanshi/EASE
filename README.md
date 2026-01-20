# EASE
The source code of the paper "EASE: Practical and Efficient Safety Alignment for Small Language Models", in the 40th Annual AAAI Conference on Artificial Intelligence (AAAI 2026)

This repository contains the official implementation of **EASE**, a two-phase safety alignment framework for small language models (SLMs)

EASE enables small language models to **selectively activate safety reasoning** for adversarial jailbreak queries while preserving **low inference overhead** for benign and straightforward harmful queries.

---

## 🔍 Overview

Small language models (SLMs) are widely deployed on edge and resource-constrained devices, but they are particularly vulnerable to jailbreak attacks. Existing alignment approaches face a trade-off:

- **Refusal training** → efficient but shallow, weak against adversarial jailbreaks  
- **Deliberative alignment** → robust but computationally expensive  

**EASE** bridges this gap via a two-phase design:

1. **Safety Reasoning Capability Implantation**  
   Distills safety reasoning patterns from a large *reasoning* teacher into an SLM.

2. **Safety Reasoning Boundary Calibration**  
   Trains the SLM to selectively apply reasoning *only* to vulnerable semantic regions where shallow alignment fails.

This results in:

- Up to **17% lower jailbreak success rate** vs. refusal training  
- Up to **90% lower inference overhead** vs. deliberative alignment  
- Near-zero degradation on general task performance (MMLU, GSM8K, HellaSwag)
