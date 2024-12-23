CIFAR-10 Image Classification with CNN and Data Augmentation:
This project implements a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset, which consists of 10 categories such as airplanes, cars, birds, and more. The model is trained using Keras with TensorFlow backend, and data augmentation techniques are applied to enhance performance.

Project Overview:
The CIFAR-10 dataset contains 60,000 32x32 color images in 10 categories, with 50,000 images for training and 10,000 for testing. This project includes:

Building a CNN model with layers such as convolution, max pooling, dropout, and dense layers.
Applying data preprocessing and normalization.
Using data augmentation to improve model generalization.
Visualizing predictions, confusion matrices, and augmented images.
Saving the trained model for future use.
Files Included
Project Code: Python code implementing the CNN model, training, evaluation, and data augmentation.
Saved Model: Trained models saved as keras_cifar10_trained_model.h5 and keras_cifar10_trained_model_Augmentation.h5.
Visualization Outputs: Confusion matrix and sample predictions visualized using Matplotlib and Seaborn.

How It Works

Data Preprocessing:

The CIFAR-10 dataset is loaded and split into training and testing sets.
Images are normalized by dividing pixel values by 255.
Labels are one-hot encoded for categorical classification.
Model Architecture:

A CNN is constructed using Keras, with layers including:
Conv2D: To extract spatial features.
MaxPooling2D: To down-sample feature maps.
Dropout: To prevent overfitting.
Dense: Fully connected layers for classification.
The final output layer uses a softmax activation function to predict the probabilities of 10 classes.
Data Augmentation:

Images are augmented with techniques like rotation, flipping, brightness adjustments, and shifting.
The augmented images are used to train the CNN, improving robustness.
Training:

The model is compiled with the RMSprop optimizer and trained using a batch size of 32.
Data augmentation is applied during training using ImageDataGenerator.
Evaluation:

The model's performance is evaluated on the test set, and the accuracy is displayed.
Predictions are visualized, showing true labels and predicted classes.
Saving the Model:

The trained model is saved as an HDF5 file for reuse or deployment.

Results

The model achieves high accuracy on the test set.
Data augmentation significantly improves the generalization of the model.
Visualization of predictions and augmented images provides insights into the model's performance.
