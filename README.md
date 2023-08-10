# AML_Final_Project
## Image Captioning Project

This repository contains an implementation of an image captioning project using the DenseNet201 architecture for image feature extraction and a tokenizer for generating captions. Image captioning is the task of generating natural language descriptions for images, enabling a computer to understand and describe the content of images.

## Overview
Image captioning is achieved through a two-step process: image feature extraction and caption generation. In this project, we utilize the DenseNet201 architecture to extract meaningful features from input images. These features are then used as context information for generating captions using a tokenizer.

## Prerequisites
Before running the project, ensure you have the following prerequisites:

Python 3.x
TensorFlow
Keras
Pillow (PIL)
NumPy
Tokenizer library

## Usage
Clone the repository and navigate to the project directory.

Place your images in the images/ directory or specify the image path in the main.py script.

Run the main.py script:


Results
After running the script, an HTML page will be available for the user to upload image. Once the image is submitted you will find the generated captions for the input images in the HTML page. 


References
DenseNet201: https://keras.io/api/applications/densenet/
