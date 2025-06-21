# E-Waste Classification Internship Project

This project implements an efficient E-Waste image classification pipeline using only the provided dataset, with state-of-the-art tricks to maximize accuracy and speed:

- **EfficientNetV2B0 backbone** + custom Squeeze-and-Excitation (SE) block  
- **MixUp data augmentation** for stronger regularization  
- **Focal Loss** to focus on hard examples  
- **Cosine-annealing learning-rate schedule** for faster convergence  
- **Knowledge Distillation**: train a lightweight student from a frozen teacher  
- **Model Pruning** (50% sparsity) + **TFLite Quantization** for low-latency inference  
- **Interactive Gradio app** for live demos

## Setup

1. Place the dataset under  
   `data/E-Waste classification dataset/train/`,  
   `…/val/`,  
   `…/test/`.  
2. Create & activate a virtualenv, then:
   ```bash
   pip install -r requirements.txt 
   python main.py
   ```