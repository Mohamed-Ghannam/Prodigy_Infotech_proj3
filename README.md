# # Hand Gesture Recognition Project

## Overview
This project implements a hand gesture recognition system using near-infrared images acquired from the Leap Motion sensor. The primary goal is to classify different hand gestures using a Convolutional Neural Network (CNN). 

## Project Workflow

1. **Libraries Used**  

   - `os`, `zipfile`, `shutil`, `numpy`, `PIL` (for image processing)

   - `matplotlib`, `seaborn` (for visualization)
   
   - `tensorflow` (for building and training the CNN model)

2. **Data Extraction and Structure**  

   The dataset consists of images of hand gestures stored in subdirectories. After unzipping and organizing the dataset, the data structure was printed to show the classes of gestures.

3. **Data Visualization: "Become One With The Data"**  

   Visualization is key to understanding the data. We displayed 3 random images for each unique gesture class, helping us to grasp the variations and features of each gesture type.

4. **Data Preprocessing**  

   We divided the dataset into training, validation, and test sets using `tensorflow.keras.preprocessing.image_dataset_from_directory` to load the images. The images were normalized and resized to (240, 600) to match the model's input shape.

5. **Model Creation: CNN Architecture**  

   We constructed a CNN with the following layers:

   - 3 Conv2D layers with MaxPooling for downsampling

   - 1 Flatten layer

   - 2 Fully connected Dense layers for classification

   The model summary was printed to understand the architecture.

6. **Model Training**  

   We compiled the model using `sparse_categorical_crossentropy` as the loss function and `adam` optimizer. The model was then trained on the training data, and the loss and accuracy were monitored for both the training and validation sets.

7. **Performance Evaluation**  

   After training, the model was evaluated on the test set, and the following were presented:

   - Loss and Accuracy

   - A classification report to summarize precision, recall, and F1-score for each gesture class

   - Confusion matrix to visualize misclassifications

8. **Prediction Function**  

   A custom function was built to predict and visualize the gesture for a random image from the test set. It displays the image, the true label, and the predicted label.

## Results

- The model's final accuracy and loss metrics on the test set were reported.

- The confusion matrix and classification report provided detailed insights into the model's performance on each gesture class.


## Conclusion

This project demonstrates the use of deep learning (CNN) for hand gesture recognition using infrared images. The visualization of the data and evaluation metrics shows the model's potential for practical applications in human-computer interaction.

