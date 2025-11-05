# Gemma-3-1B Tamil Nadu Cultural AI Model Sample

[![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/naazimsnh02/gemma-3-tamilnadu_sample)
[![License](https://img.shields.io/badge/License-Gemma-blue.svg)](https://ai.google.dev/gemma/terms)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A fine-tuned version of **Gemma-3-1B-IT** specifically trained on **Tamil Nadu personas and cultural context**. This model understands and responds with authentic Tamil Nadu cultural knowledge, mixing Tamil and English naturally, and demonstrates deep understanding of Tamil traditions, festivals, language, and regional nuances.

🔗 **[Model on Hugging Face](https://huggingface.co/naazimsnh02/gemma-3-tamilnadu_sample)**

---

## ✨ Key Features

- 🎯 **Tamil Nadu Focused**: Trained exclusively on personas from Tamil Nadu
- 🗣️ **Bilingual**: Natural code-mixing of Tamil and English
- 🎭 **Cultural Awareness**: Deep understanding of Tamil festivals, traditions, and customs
- 🛡️ **Safety Aligned**: Includes safety fine-tuning to refuse harmful requests
- 📚 **Multi-Stage Training**: SFT → Instruction Tuning → DPO → Safety

---

## 📊 Training Details

### Training Data

- **Primary Dataset**: Nemotron-Personas-India (en_IN split, Tamil Nadu only)
- **Personas**: ~20,000 Tamil Nadu personas
- **Instructions**: Tamil Nadu-specific + general knowledge mix
- **DPO Pairs**: Cultural preference alignment
- **Safety Examples**: Tamil Nadu context-aware safety responses

### Training Pipeline

| Stage | Dataset | Steps | Focus |
|-------|---------|-------|-------|
| **Stage 1: SFT** | Tamil Nadu personas from Nemotron-Personas-India | 3000 | Learning to roleplay as diverse Tamil Nadu personas |
| **Stage 2: Instruction Tuning** | Tamil Nadu instructions + general knowledge (80/20 mix) | 500 | Task-specific instruction following |
| **Stage 3: DPO** | Tamil Nadu preference pairs (Beta: 0.1) | 150 | Aligning responses with Tamil cultural preferences |
| **Stage 4: Safety** | Safety examples with Tamil Nadu context | 50 | Ethical responses, anti-discrimination, cultural sensitivity |

### Training Configuration

```yaml
Base Model: unsloth/gemma-3-1b-it
LoRA Rank: 32
LoRA Alpha: 32
Max Sequence Length: 2048
Quantization: 4-bit
Training Framework: Unsloth + TRL
GPU: NVIDIA T4 (Google Colab)
Training Time: ~10 hours (full pipeline)
```

---

## 🚀 Quick Start

### Installation

```bash
pip install transformers torch unsloth
```

### Basic Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "naazimsnh02/gemma-3-tamilnadu_sample"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Example prompt
messages = [
    {"role": "user", "content": "Tell me about Pongal festival in Tamil Nadu"}
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to("cuda")

outputs = model.generate(
    inputs,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### With Unsloth (Faster Inference)

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="naazimsnh02/gemma-3-tamilnadu_sample",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)

# Generate response
messages = [{"role": "user", "content": "Vanakkam! How are you?"}]
inputs = tokenizer.apply_chat_template(
    messages, 
    tokenize=True, 
    add_generation_prompt=True, 
    return_tensors="pt"
).to("cuda")

outputs = model.generate(inputs, max_new_tokens=256, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## ⚠️ Limitations

- **Language**: Primarily English with Tamil code-mixing; not fluent in pure Tamil text generation
- **Geographic Scope**: Optimized for Tamil Nadu context; may not generalize well to other Indian states
- **Persona Bias**: Training focused on specific demographic distributions from the dataset
- **Cultural Nuances**: May not capture all sub-regional variations within Tamil Nadu
- **Safety**: While safety-tuned, may still generate inappropriate content in edge cases

---

## 🤝 Ethical Considerations

- **Anti-Discrimination**: Model is trained to reject caste-based, religious, or regional discrimination
- **Cultural Sensitivity**: Respects Tamil Nadu's diverse communities and traditions
- **Bias Mitigation**: Includes safety fine-tuning to promote equality and respect
- **Responsible Use**: Should not be used to generate harmful, discriminatory, or misleading content

---

## 📚 Citation

If you use this model, please cite:

```bibtex
@misc{tamil-nadu-gemma-2025,
  title={Tamil Nadu Cultural AI Model based on Gemma-3-1B-IT},
  author={Syed Naazim Hussain},
  year={2025},
  publisher={HuggingFace},
  howpublished={\url{https://huggingface.co/naazimsnh02/gemma-3-tamilnadu_sample}}
}
```

---

## 🙏 Acknowledgments

- **Base Model**: Google's Gemma-3-1B-IT via Unsloth
- **Dataset**: NVIDIA's Nemotron-Personas-India
- **Framework**: Unsloth for efficient training
- **Community**: Tamil Nadu's rich cultural heritage

---

## 📄 License

This model inherits the Gemma license from the base model. Please review the [license terms](https://ai.google.dev/gemma/terms) before use.

---

## 📧 Contact

For questions, issues, or feedback, please open an issue on this repository or visit the [model page on Hugging Face](https://huggingface.co/naazimsnh02/gemma-3-tamilnadu_sample).

---

**Note**: This model is designed for research and educational purposes. Always verify outputs for accuracy and cultural appropriateness.
