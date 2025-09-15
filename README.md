# Self-Supervised Card Classification

## Overview
This project implements a self-supervised learning pipeline for card type classification using PyTorch. The workflow consists of two main phases:

1. **Self-Supervised Pretraining**: The model learns robust visual features by predicting the rotation angle of card images (0°, 90°, 180°, 270°) in an unsupervised manner.
2. **Supervised Fine-Tuning**: The pretrained encoder is fine-tuned on a small labeled dataset to classify card types (e.g., bank card, ID card, visiting card, voter ID).

A Streamlit web app is provided for easy card type prediction using the trained model.

---

## Project Structure
```
├── app.py                        # Streamlit app for card classification
├── Card_Classification_SelfSupervised.ipynb  # Full training and evaluation notebook
├── card_classifier.pth           # Trained classifier weights
├── cards_labeled_small/          # Labeled images (subfolders per class)
│   ├── bank_card/
│   ├── id_card/
│   ├── visiting_card/
│   └── voter_id/
├── my_card_images/               # Unlabeled images for self-supervised pretraining
```

---

## Getting Started

### 1. Environment Setup
- Python 3.8+
- Install dependencies:
  ```bash
  pip install torch torchvision pillow streamlit
  ```

### 2. Data Preparation
- Place unlabeled card images in `my_card_images/` for self-supervised training.
- Organize labeled images in `cards_labeled_small/` with one subfolder per class (e.g., `bank_card`, `id_card`, etc.).

---

## Training Pipeline

### Self-Supervised Pretraining
- Run the notebook `Card_Classification_SelfSupervised.ipynb` or use the provided Colab badge to train the encoder on rotation prediction.
- The encoder weights are saved as `self_supervised_encoder.pth`.

### Supervised Fine-Tuning
- Continue in the notebook to fine-tune the classifier on the labeled dataset.
- The final model is saved as `card_classifier.pth`.

---

## Inference with Streamlit

1. Ensure `card_classifier.pth` is in the project root.
2. Run the app:
   ```bash
   streamlit run app.py
   ```
3. Upload a card image to get its predicted type.

---

## Model Architecture
- **Encoder**: Simple CNN with two convolutional layers and max pooling.
- **Classifier**: Fully connected layers for card type prediction.

---

## Example Classes
- `bank_card`
- `id_card`
- `visiting_card`
- `voter_id`

---

## References
- [PyTorch Documentation](https://pytorch.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Self-Supervised Learning Papers](https://arxiv.org/abs/1805.01978)

---

## License
MIT License
