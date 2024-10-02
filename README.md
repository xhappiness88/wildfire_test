### README.md

# Wildfire Classification Using Satellite Images

## Overview
This project aims to develop a deep learning model to classify satellite images into two categories: "Wildfire" and "No Wildfire." The model leverages the EfficientNet_B0 architecture, known for its efficiency and accuracy in image classification tasks. The project includes data preprocessing, model training, evaluation, and deployment through a web interface built with Streamlit.

## Project Structure
The workspace is organized as follows:

```
1.py
app.py
demo.ipynb
going_modular/
	__pycache__/
	data_setup.py
	engine.py
	model_builder.py
	predictions.py
	README.md
	train.py
	utils.py
images/
model/
	EfficientNet_b0-Wildfire_Classifier.pt
pages/
	Use Custom Images.py
	Use Validation Images.py
Read.me
readme.md
requirements.txt
validation_dataset/
	README.txt
	valid/
		nowildfire/
		wildfire/
wildfire-classification.ipynb
WildFire.pptx
```

## Key Components

### Model Selection and Preprocessing
- The project employs a pretrained EfficientNet_B0 model from the torchvision library.
- Input images are normalized and resized using transformations derived from the EfficientNet_B0 pretrained weights.

### Data Loading
- Data loaders are created for training and validation datasets, facilitating efficient batch processing.

### Training
- The model is trained on a labeled dataset of satellite images.
- The training process involves adjusting the model's weights using an optimizer and a loss function.

### Evaluation
- The model's performance is evaluated on a separate validation dataset to assess its accuracy in classifying images.

### Model Saving and Loading
- The trained model is saved to disk and can be loaded back for classification tasks or further fine-tuning.
- Example: The model is saved as `EfficientNet_b0-Wildfire_Classifier.pt` in the [`model`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FD%3A%2FGinnyPig%2FWildfire%2Fmodel%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "d:\GinnyPig\Wildfire\model") directory.

### Web Interface
- A Streamlit web interface allows users to interact with the model.
- Users can upload images for classification and view the results.

## Notable Files

- **[`wildfire-classification.ipynb`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FD%3A%2FGinnyPig%2FWildfire%2Fwildfire-classification.ipynb%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "d:\GinnyPig\Wildfire\wildfire-classification.ipynb")**: Jupyter notebook detailing the entire workflow from data preprocessing to model evaluation.
- **[`going_modular/data_setup.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FD%3A%2FGinnyPig%2FWildfire%2Fgoing_modular%2Fdata_setup.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "d:\GinnyPig\Wildfire\going_modular\data_setup.py")**: Script for preparing and downloading the dataset.
- **[`going_modular/engine.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FD%3A%2FGinnyPig%2FWildfire%2Fgoing_modular%2Fengine.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "d:\GinnyPig\Wildfire\going_modular\engine.py")**: Contains the core training loop and evaluation functions.
- **[`going_modular/model_builder.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FD%3A%2FGinnyPig%2FWildfire%2Fgoing_modular%2Fmodel_builder.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "d:\GinnyPig\Wildfire\going_modular\model_builder.py")**: Script for creating the PyTorch model.
- **[`going_modular/predictions.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FD%3A%2FGinnyPig%2FWildfire%2Fgoing_modular%2Fpredictions.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "d:\GinnyPig\Wildfire\going_modular\predictions.py")**: Utility functions for making predictions with the trained model.
- **[`app.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FD%3A%2FGinnyPig%2FWildfire%2Fapp.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "d:\GinnyPig\Wildfire\app.py")**: Streamlit application for interacting with the model.

## Dataset
- The dataset used for training and validation is sourced from Kaggle, specifically designed for wildfire prediction tasks.
- The dataset is organized into "wildfire" and "nowildfire" categories within the [`validation_dataset/valid/`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FD%3A%2FGinnyPig%2FWildfire%2Fvalidation_dataset%2Fvalid%2F%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "d:\GinnyPig\Wildfire\validation_dataset\valid\") directory.
- Dataset link: [Wildfire Prediction Dataset](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset)

## Requirements
The project requires the following Python packages, as specified in the [`requirements.txt`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FD%3A%2FGinnyPig%2FWildfire%2Frequirements.txt%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "d:\GinnyPig\Wildfire\requirements.txt") file:

```txt
torch==1.12.1
torchvision==0.13.1
torchinfo
pathlib
tqdm
streamlit==1.27.0
pillow==9.4.0
matplotlib
```

## Running the Project

### Training the Model
1. Ensure you have the required packages installed:
   ```sh
   pip install -r requirements.txt
   ```
2. Run the training script:
   ```sh
   python going_modular/train.py
   ```

### Using the Web Interface
1. Run the Streamlit app:
   ```sh
   streamlit run app.py
   ```
2. Open the provided URL in your web browser to interact with the model.

## Conclusion
This project exemplifies the application of deep learning in environmental monitoring, showcasing how satellite imagery can be used to detect and classify wildfires, potentially aiding in early detection and response efforts. The benefits include improved public safety, cost reduction, scalability, and global coverage, making it a valuable tool in the fight against wildfires.
