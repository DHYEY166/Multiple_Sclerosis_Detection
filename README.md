##Breast Cancer Semantic Segmentation

Welcome to my Data Science Project! This project focuses on performing breast cancer semantic segmentation using a deep learning model. The application is built with Streamlit, allowing users to upload images and receive segmented outputs indicating different classes related to breast cancer tissues.

- [App Overview](#app-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Dataset](#dataset)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## App Overview

This project involves classifying MRI images into categories related to Multiple Sclerosis (MS) using a VGG16-based Convolutional Neural Network (CNN). The dataset used for this project is sourced from Kaggle, specifically designed for MS image classification. The application allows users to upload an MRI image, and it will process the image using a pre-trained deep learning model to classify it into categories such as Control-Axial, Control-Sagittal, MS-Axial, and MS-Sagittal.

## Installation

- Make sure you have the necessary libraries installed. You can install them using pip:

   pip install numpy pandas matplotlib tensorflow scikit-learn keras pillow opencv-python plotly

- To run this application locally, follow these steps:

  1. Clone the repository:
   
   git clone https://github.com/DHYEY166/Multiple_Sclerosis_Detection.git
   
   cd multiple_sclerosis_detection

  2. Create a virtual environment (optional but recommended):

   python -m venv venv
   
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

  3. Install the required packages:

   pip install -r requirements.txt

  4. Run the Streamlit application:

   streamlit run ms_git.py

## Usage

- **Data Augmentation**: The code includes a data augmentation script that increases the number of images in each category to a desired number (3000 in this case). This helps improve the model's performance by providing more training data.
- **Model Training**: The VGG16 model is loaded with pre-trained weights and fine-tuned for the classification task. The model is compiled and trained using the augmented dataset.
- **Evaluation**: After training, the model's performance is evaluated using accuracy, classification reports, and ROC curves.
- **Predictions**: The model is used to make predictions on new MRI images. The code includes functionality to load and preprocess these images for prediction.

You can also access the application directly via the following link:

[Streamlit Application](https://multiplesclerosisdetection-dwulhuy4hrgstvbyg5mqht.streamlit.app)

## Model Details

The primary objective of this project is to build a robust image classification model that can accurately classify MRI images into one of the following categories:

Control-Axial
Control-Sagittal
MS-Axial
MS-Sagittal
The VGG16 model is utilized for feature extraction, and additional layers are added for classification. Data augmentation is applied to enhance the training dataset.

## Dataset

The dataset used for this project can be found on Kaggle: [Multiple Sclerosis Dataset](https://www.kaggle.com/datasets/buraktaci/multiple-sclerosis)

## Features

**Streamlit Application**: A Streamlit application is available for interactive use. You can upload an MRI image to classify it into one of the categories. The application provides an easy-to-use interface for testing the model with new images.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/DHYEY166/Multiple_Sclerosis_Detection/blob/main/LICENSE) file for more details.

## Contact

- **Author**: Dhyey Desai
- **Email**: dhyeydes@usc.edu
- **GitHub**: https://github.com/DHYEY166
- **LinkedIn**: https://www.linkedin.com/in/dhyey-desai-80659a216 

Feel free to reach out if you have any questions or suggestions.
