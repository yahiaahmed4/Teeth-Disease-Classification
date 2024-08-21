# Teeth Diseases Classification with CNN

This project aims to classify seven types of teeth diseases using a Convolutional Neural Network (CNN) built from scratch using TensorFlow. The model is trained on a custom dataset, which includes images categorized into seven distinct classes representing different teeth conditions.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Structure](#dataset-structure)
- [Installation](#installation)
- [Data Loading](#data-loading)
- [Data Visualization](#data-visualization)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)
- [Saving and Loading the Model](#saving-and-loading-the-model)
- [Inference](#inference)
- [Results](#results)

## Project Overview
This project involves building a CNN model to classify images of teeth into one of seven disease categories. The model is trained using TensorFlow, and various techniques such as data augmentation and early stopping are applied to improve its performance. The final model is saved for future inference tasks.

## Dataset Structure
The dataset is organized into three main directories:
- `Training`: Contains images used for training the model.
- `Validation`: Contains images used for validating the model during training.
- `Testing`: Contains images used to evaluate the model after training.

Each directory has subfolders named after the seven disease classes:
- **CaS**: Candidiasis
- **CoS**: Composite Restorations
- **Gum**: Gingivitis
- **MC**: Mouth Cancer
- **OC**: Oral Cysts
- **OLP**: Oral Lichen Planus
- **OT**: Oral Trauma

## Installation
To run this project, you need to have Python installed along with the following libraries:
```bash
pip install tensorflow matplotlib scikit-learn
```

## Data Loading
The images are loaded from their respective directories using TensorFlow’s `image_dataset_from_directory` function. This function allows the images to be batched, resized to a uniform size (224x224), and labeled according to their directory.

```python
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'Teeth_Dataset/Training',
    image_size=(224, 224),
    batch_size=32,
    label_mode='int'
)
```

### Explanation
- **Batching**: Images are loaded in batches of 32 to optimize memory usage.
- **Resizing**: All images are resized to 224x224 pixels to maintain consistency across the dataset.
- **Labeling**: Labels are automatically assigned based on the directory names.

## Data Visualization
Visualizing the data helps in understanding the distribution and types of images within the dataset. Sample images can be plotted along with their corresponding labels.

### Explanation
A function is provided to visualize a grid of sample images from the dataset, displaying the diversity of data and ensuring that the data loading process is correct.

## Data Preprocessing
### Normalization
Before feeding the images into the model, they are normalized. Normalization scales the pixel values to the range `[0, 1]` to facilitate faster convergence during training.

```python
def normalize_dataset(dataset):
    return dataset.map(lambda x, y: (x / 255.0, y))

train_ds = normalize_dataset(train_ds)
```

### Data Augmentation
To improve the model's generalization ability, data augmentation is applied. This includes random flipping, rotation, zooming, and contrast adjustments.

```python
data_augmentation = tf.keras.Sequential([...])
```

### Explanation
- **Normalization**: Converts pixel values from `[0, 255]` to `[0, 1]`.
- **Data Augmentation**: Introduces variations in the training data to help the model learn to generalize better.

## Model Architecture
The CNN model is composed of several convolutional and max-pooling layers, followed by dense layers. The architecture is designed to extract features from the images and classify them into one of the seven disease categories.

```python
model = models.Sequential([...])
```

### Explanation
- **Convolutional Layers**: Extract features from the images using filters.
- **Max-Pooling Layers**: Reduce the spatial dimensions, helping to make the model invariant to small translations in the input.
- **Dense Layers**: Perform classification based on the extracted features.

## Training the Model
The model is trained using the Adam optimizer and sparse categorical cross-entropy loss. Early stopping is employed to prevent overfitting by halting the training when the validation loss stops improving.

```python
early_stopping = EarlyStopping([...])
history = model.fit([...])
```

### Explanation
- **Adam Optimizer**: An adaptive learning rate optimizer that is well-suited for training deep neural networks.
- **Early Stopping**: Stops training when further improvements in validation loss are not observed, helping to avoid overfitting.

## Evaluation
After training, the model is evaluated on the test dataset. The evaluation metrics include accuracy and loss, along with a confusion matrix and classification report to assess the model’s performance on each class.

```python
test_loss, test_acc = model.evaluate(test_ds)
```

### Explanation
- **Confusion Matrix**: Shows how well the model is performing in terms of correctly and incorrectly classified instances across all classes.
- **Classification Report**: Provides precision, recall, and F1-score for each class.

## Saving and Loading the Model
The trained model is saved as an HDF5 file, which can be reloaded for future use without the need for retraining.

```python
model.save("TeethClassification.h5")
```

### Explanation
- **Saving the Model**: Saves the entire model architecture, weights, and optimizer state, allowing the model to be reloaded and used for inference or further training.

## Inference
To make predictions on new data using the saved model, the image must be preprocessed in the same way as the training data. This includes resizing, normalization, and converting the image to a format that the model can process.

### Step-by-Step Guide:
1. **Load the Image**: Load the image from the file path and resize it to `(224, 224)`.
2. **Normalize the Image**: Scale the pixel values to the `[0, 1]` range.
3. **Expand Dimensions**: Add a batch dimension to the image array.
4. **Prediction**: Use the loaded model to predict the class.

```python
img = image.load_img('infere_img2.jpg', target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)
predictions = loaded_model.predict(img_array)
```

### Explanation
- **Image Preprocessing**: Ensures that the input image has the same format as the data used during training.
- **Batch Dimension**: Even if predicting a single image, it must be provided as a batch to the model.

## Results
- **Training Accuracy**: Approximately 95.69%
- **Validation Accuracy**: Approximately 87.65%
- **Test Accuracy**: Approximately 87.65%

The model demonstrates high accuracy in classifying teeth diseases, with strong generalization to unseen data. The confusion matrix and classification report provide further insights into the model's performance across the different disease categories.

