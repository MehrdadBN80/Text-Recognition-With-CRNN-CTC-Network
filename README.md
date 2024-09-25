# CAPTCHA Recognition using CRNN

This repository contains a PyTorch implementation for recognizing text from CAPTCHA images using a Convolutional Recurrent Neural Network (CRNN) model. The model uses a combination of CNNs for feature extraction and RNNs for sequence modeling to predict the CAPTCHA text from input images.

## Table of Contents

1. [Dataset](#dataset)
2. [Model Architecture](#model-architecture)
3. [Training Process](#training-process)
4. [Evaluation](#evaluation)
5. [Usage](#usage)
6. [Results](#results)
7. [Saving and Loading Model](#saving-and-loading-model)

## Dataset

The dataset consists of CAPTCHA images, where each image file name represents the ground truth text. The images are grayscale PNG files, and the text includes lowercase English letters (`a-z`) and digits (`0-9`). Each CAPTCHA image represents a sequence of 5 characters.

**Example of the dataset structure:**
- `6n6gg.png`: Ground truth label is `6n6gg`
- `fncnb.png`: Ground truth label is `fncnb`

The dataset is split into training and testing sets, using an 80-20 split.

## Model Architecture

The model follows the Convolutional Recurrent Neural Network (CRNN) architecture:

1. **CNN (Feature Extraction):**
   - A series of convolutional layers extract features from the CAPTCHA images.
   - The CNN is composed of multiple convolutional, ReLU, and BatchNorm layers.
   
2. **RNN (Sequence Modeling):**
   - Bidirectional LSTM (or GRU) layers are used to model the sequential dependencies in the extracted features.

3. **CTC Loss:** 
   - Connectionist Temporal Classification (CTC) loss is used during training to handle variable-length sequences and alignment between predictions and labels.

### Key Layers:
- **Convolutional Layers:** Extract features from the image.
- **Bidirectional LSTM (or GRU):** Processes the sequential data from the CNN output.
- **CTC Loss:** Handles alignment between input and target sequences.

## Training Process

The training is performed using the Adam optimizer and CTC loss function. The model is trained for a default of 50 epochs on the training set, with real-time tracking of the loss:

1. **Data Loading:**
   - A custom dataset class `CaptchaDataset` is used to load and preprocess CAPTCHA images and their corresponding labels.

2. **Training Loop:**
   - The `Engine` class is responsible for training the model and managing the optimizer steps, loss calculation, and backpropagation.

3. **Early Stopping:** (Optional)
   - The training process can include early stopping based on validation performance to avoid overfitting.

### Training Example:

```python
engine.fit(train_loader)
```

## Evaluation

During evaluation, the model predicts the sequences from the test data and calculates the loss. The evaluation is performed using the same `Engine` class.

### Evaluation Example:

```python
outs, loss = engine.evaluate(test_loader)
```

The output predictions are compared with the ground truth to assess model performance.

## Usage

To use the trained model for inference, simply pass a CAPTCHA image through the `predict` function. The function converts the image to tensor format, passes it through the model, and outputs the predicted text.

### Example Code for Prediction:

```python
image = data[ids]  # Random image from the dataset
out = engine.predict(image)[0]
show_prediction(out, image)
```

The `show_prediction` function displays the image with the predicted text alongside the ground truth.

### Image Prediction Example:

```python
show_prediction(out, image)
```

## Results

The model shows an improvement over time as the loss decreases across epochs, demonstrating its ability to learn the CAPTCHA recognition task. Example loss values across training epochs:

```
Epoch 1: Loss = 4.32
Epoch 10: Loss = 3.29
Epoch 30: Loss = 0.95
Epoch 50: Loss = 0.156
```

## Saving and Loading Model

After training, the model's state, optimizer state, and mappings are saved to a file. You can load the trained model for later inference or fine-tuning.

### Saving Model:

```python
saving = {'state_dict': engine.model.state_dict(),
          'optimizer': engine.optimizer.state_dict(),
          'mapping': mapping,
          'mapping_inv': mapping_inv}
torch.save(saving, './model.pth')
```

### Loading Model:

```python
checkpoint = torch.load('./model.pth')
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])
```

## Dependencies

- Python 3.x
- PyTorch
- torchvision
- pandas
- Pillow
- tqdm
- scikit-learn

Install required libraries using:

```bash
pip install torch torchvision pandas pillow tqdm scikit-learn
```

## Conclusion

This project implements a powerful CRNN model to solve CAPTCHA recognition tasks. By leveraging both convolutional layers for feature extraction and recurrent layers for sequence prediction, the model achieves reliable results on CAPTCHA text recognition.
