
# Musical Instrument Classification : PretraineCNN + ML Pipeline


## Project Overview

This repository contains a two-stage hybrid model for classifying images of musical instruments. The pipeline combines deep learning feature extraction with traditional machine learning classification to leverage the strengths of both approaches.




![App Screenshot](https://raw.githubusercontent.com/bhaktii1802/Hybrid_models/main/pngtree-3d-rendered-musical-instruments-image_3886080.jpg)


## Architecture and Pipeline Steps

#### Data Loading: 

Images of musical instruments are organized into train and test directories by class labels.

ImageDataGenerator is used for on-the-fly image loading and basic preprocessing (rescaling).

#### Feature Extraction (Deep CNN):

A pretrained ResNet50 model (excluding the top classification layer) is used as a fixed feature extractor.

Global average pooling reduces the convolutional feature maps into a 2048-dimensional feature vector per image.

Extracted features for train and test sets are saved as NumPy arrays.

#### Feature Preprocessing:

StandardScaler normalizes the features to zero mean and unit variance.

PCA reduces dimensionality while retaining ~98% of the variance, speeding up training and reducing overfitting.

#### Machine Learning Classification:

A Logistic Regression model is trained on the PCA-transformed training features.

The model predicts instrument labels on the test set features.

#### Evaluation:

Classification performance is measured using a classification report (precision, recall, F1-score) and a confusion matrix.

Results are visualized to identify misclassifications and guide potential improvements.

Inference on New Images

A utility function allows inference on individual images: loads image, extracts features, applies scaling & PCA, and outputs predicted label.




## Installation
git clone https://github.com/<your-username>/musical-hybrid.git
cd musical-hybrid

#### Create and activate virtual environment
```
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\\Scripts\\activate    # Windows
```

#### Install dependencies
```
pip install -r requirements.txt
```
## Usage/Examples
####  Extract Features
```javascript
python extract_features.py --train_dir data/train --test_dir data/test --output_dir features/
```
#### Train Classifier
```javascript
python train_classifier.py --feature_dir features/ --model_dir models/
```
#### Evaluate
```javascript
python evaluate.py --model_dir models/ --feature_dir features/
```

####  Predict
```javascript
python predict.py --model_path models/logreg.pkl --image_path sample.jpg

```


### Results

Achieved XX% accuracy on the test set.

Example classification report and confusion matrix are saved under reports/.
