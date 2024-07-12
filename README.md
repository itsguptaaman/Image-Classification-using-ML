# Image Classification Using Machine Learning

## Overview

The Image Classification Using Machine Learning project aims to classify images into predefined categories using conventional machine learning techniques. The primary components of this project include data preprocessing, feature extraction, dimensionality reduction, model training, evaluation, and deployment via a Flask web application.

## Table of Contents
- [Overview](#overview)
- [Purpose](#purpose)
- [Key Features](#key-features)
- [Use Cases](#use-cases)
- [Technology Stack](#technology-stack)
- [Getting Started](#getting-started)
- [Testing](#testing)
- [Contributing](#contributing)

## Purpose

The main objective of this project is to provide accurate image classification without relying on deep learning models. It categorizes images into six distinct classes: Building, Forest, Glacier, Mountain, Sea, and Street, utilizing traditional machine learning techniques.

## Key Features
- **Data Preprocessing:** Organizes and preprocesses the dataset, including image resizing and normalization.
- **Feature Extraction:** Utilizes handcrafted features to represent images.
- **Dimensionality Reduction:** Optionally reduces the feature space for improved efficiency.
- **Model Training:** Trains various shallow learning models on the extracted features.
- **Model Evaluation:** Assesses model performance using appropriate metrics.
- **Deployment:** Implements a Flask web application for easy image classification.

## Use Cases
- **Image Classification:** Categorize images into predefined classes for various applications.
- **Content Analysis:** Analyze and classify images for content moderation, product recognition, and visual search.
- **Educational Purposes:** Serve as a learning tool for understanding traditional machine learning techniques in image classification.

## Technology Stack
- **Python:** Core programming language for the project.
- **Flask:** Web framework for deploying the image classification model.
- **Scikit-learn:** Library for implementing traditional machine learning models.
- **NumPy & Pandas:** Libraries for data manipulation and preprocessing.
- **OpenCV:** Library for image processing.

## Getting Started

### Requirements
Please refer to the `installation_steps.md` file for detailed installation instructions.

### Running the Flask Application

#### 1. Start the Flask server
```bash
python app.py
```

#### 2. Access the web interface
Open your browser and go to http://127.0.0.1:5000 to upload images and get predictions.


### Testing
Sample images for testing the model are provided in the test_images directory. Use these images to validate the model's performance via the Flask application.

### Contributing
Contributions are welcome! Please fork this repository and submit a pull request with your enhancements.