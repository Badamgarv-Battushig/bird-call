# Neural Networks for Seattle-Area Bird Call Classification

This project uses neural networks to classify Seattle-area bird species based on their sound spectrograms. The main goal is to test whether a custom convolutional neural network can learn visual patterns in bird-call spectrograms and use them to predict bird species.

The project was completed for **DATA 5322 Statistical Machine Learning 2**.

---

## Project Overview

Bird calls can be represented as spectrograms, where time, frequency, and sound intensity are shown as an image. Because the input data are image-like spectrograms, convolutional neural networks are used for classification.

This project includes:

- A binary CNN model for classifying two bird species
- Multiple multiclass CNN models for classifying 12 bird species
- Model comparison using accuracy, loss, confusion matrices, precision, recall, and F1-score
- Predictions for three external mystery bird-call clips
- Discussion of limitations, model errors, and possible future improvements

---

## Research Questions

1. Can a neural network distinguish between two similar Seattle-area bird species using spectrograms?
2. How well can different CNN architectures classify all 12 bird species?
3. Which bird species are most often confused by the model?
4. Can the trained multiclass CNN make reasonable predictions on external mystery bird-call clips?

---

## Dataset

The data contain preprocessed spectrograms from 12 Seattle-area bird species. The spectrogram data are stored in an HDF5 file.

Each spectrogram represents bird-call audio as an image, where:

- The horizontal axis represents time
- The vertical axis represents frequency
- Pixel intensity represents sound energy

The 12 bird species are:

| Code | Species |
|---|---|
| amecro | American Crow |
| amerob | American Robin |
| bewwre | Bewick's Wren |
| bkcchi | Black-capped Chickadee |
| daejun | Dark-eyed Junco |
| houfin | House Finch |
| houspa | House Sparrow |
| norfli | Northern Flicker |
| rewbla | Red-winged Blackbird |
| sonspa | Song Sparrow |
| spotow | Spotted Towhee |
| whcspa | White-crowned Sparrow |

To reduce imbalance, larger classes were capped at 120 samples per species.

---

## Methods

### Data Preparation

The notebook performs the following data preparation steps:

1. Load spectrogram arrays from the HDF5 file
2. Convert the data into the format expected by Keras CNN models
3. Scale spectrogram values to a neural-network-friendly range
4. Resize spectrogram images to 64 x 160 pixels
5. Encode species labels
6. Use stratified train/test splitting for fair evaluation
7. Apply the same preprocessing procedure to the external mystery clips

### Models

The project uses custom neural networks built with TensorFlow/Keras.

The models include:

| Model | Task | Purpose |
|---|---|---|
| Binary CNN | American Robin vs Song Sparrow | Test whether a CNN can separate two bird species |
| Small CNN | 12-class classification | Baseline multiclass CNN |
| Deeper CNN | 12-class classification | Tests whether adding more convolutional depth improves performance |
| Wide CNN | 12-class classification | Tests whether using more filters improves performance |

The multiclass CNN models are compared to evaluate how model structure affects performance.

### Tuning Strategy

Model tuning was done by comparing different CNN architectures. The small CNN was used as a baseline, the deeper CNN tested additional convolutional layers, and the wide CNN tested a larger number of filters.

The models also used:

- Early stopping
- Learning-rate reduction
- Same image size
- Same batch size
- Same maximum number of epochs
- Same train/test split

This made the model comparison fair and reproducible.

---

## Evaluation Metrics

The models were evaluated using:

- Test accuracy
- Test loss
- Confusion matrix
- Precision
- Recall
- F1-score
- Training time

For the multiclass model, class-level performance was especially important because some species were more difficult to classify than others.

---

## Main Results

The binary model performed better than the multiclass models because it only needed to separate two species.

| Model | Task | Test Accuracy | Training Time |
|---|---|---:|---:|
| Binary CNN | American Robin vs Song Sparrow | 79.2% | 3.7 seconds |
| Wide CNN | 12 species | 29.6% | 36.8 seconds |
| Small CNN | 12 species | 27.5% | 8.2 seconds |
| Deeper CNN | 12 species | 10.3% | 11.4 seconds |

The wide CNN was the best multiclass model. However, the 12-species classification task was difficult because several bird calls had similar spectrogram patterns.

---

## External Test Clip Predictions

The best multiclass CNN was applied to three mystery bird-call clips.

| Clip | Top Predicted Species | More Than One Bird? |
|---|---|---|
| test1 | House Sparrow | Yes |
| test2 | Red-winged Blackbird | No |
| test3 | Black-capped Chickadee | Yes |

These predictions should be interpreted cautiously because the best multiclass model reached only 29.6% test accuracy.

---

## Project Files

| File / Folder | Description |
|---|---|
| `bird_call_neural_networks.ipynb` | Main notebook with code, models, plots, and results |
| `bird_call_neural_networks.html` | Rendered notebook output for easier viewing |
| `bird_call_classification_formal_report_plain.docx` | Formal written report |
| `outputs/figures/` | Saved figures generated by the notebook |
| `outputs/models/` | Saved model files, if included |
| `data/` | Folder for the HDF5 spectrogram file and mystery audio clips |

---

## How to Run the Notebook

1. Clone this repository.

```bash
git clone <your-repository-link>
cd <your-repository-folder>
