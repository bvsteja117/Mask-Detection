
# Face Mask Detection with TensorFlow and Keras

This project involves training a deep learning model to detect whether a person is wearing a face mask or not. The dataset used for this project is sourced from Kaggle and contains images of people with and without face masks.

## Dataset

The dataset used in this project can be found on Kaggle: [Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset/data). It contains images categorized into two classes: 
- With Mask
- Without Mask

## Project Structure

- **Libraries Imported**: The necessary libraries, including TensorFlow, Keras, and Matplotlib, are imported at the beginning. 
- **Dataset Loading**: The dataset is loaded using `tf.keras.preprocessing.image_dataset_from_directory`, and the images are shuffled and split into batches.
- **Data Visualization**: Visualizations include bar plots, pie charts, and sample images from the dataset, allowing a better understanding of the class distribution.
- **Dataset Splitting**: The dataset is split into training, validation, and test sets using a custom function.
- **Data Preprocessing**: Images are resized and rescaled using Keras layers.
- **Modeling**: A model is created using the pre-trained ResNet152V2 architecture, followed by custom dense layers for binary classification.
- **Model Training**: The model is compiled with Adamax optimizer and trained for 10 epochs.
- **Model Evaluation**: The model is evaluated on the test dataset, and accuracy and loss curves are plotted.
- **Prediction**: A function is provided to predict whether a given image contains a person wearing a mask or not, along with the confidence level.

## Files in This Repository

- **mask-detection.ipynb**: The Jupyter notebook containing all the code for data loading, preprocessing, model training, and evaluation.
- **README.md**: This file, providing an overview of the project.

## How to Use

1. **Install Dependencies**: Ensure you have Python and the necessary libraries installed. You can install the required libraries using:

   ```bash
   pip install -r requirements.txt
   ```

2. **Download Dataset**: Download the dataset from the [Kaggle link](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset/data) and place it in the appropriate directory.

3. **Run the Notebook**: Open the notebook in Jupyter or any other compatible environment and run the cells sequentially to load the dataset, train the model, and make predictions.

## Results

The trained model provides predictions on whether an individual is wearing a face mask with a confidence score. Visualization of accuracy and loss during training gives insights into the model's performance.

## Acknowledgements

- Dataset: [Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset/data) by Omkar Gurav.


