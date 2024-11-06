# Research Abstract Summarizer
This project provides a machine-learning solution for analyzing research abstracts. It includes a model pipeline and a code structure that loads, preprocesses, trains, and evaluates a model on research text data. It focuses on identifying specific sections (such as OBJECTIVE, METHODS, and RESULTS) from abstracts.

## Overview
This model leverages TensorFlow and TensorFlow Hub embeddings to predict research abstract sections. The notebook contains data loading, processing, model training, evaluation, and final testing. Additionally, it includes code to save and load the model, making it easy to replicate the evaluation and results.

## Project Structure
1. Skimlit_ML_Assignment.ipynb - A main notebook with code to run the ML pipeline, train the model and evaluate.
2. Documentation Report.
3. model.h5 - Saved model file (generated after training).

## Installation
To set up the project, clone the repository, and install the required libraries. Set up a virtual environment using bash to avoid conflicts:

```
git clone <repository_url>
cd SkimLit
python3 -m venv skimlit_env
source skimlit_env/bin/activate  
```

Then install dependencies with:

```
pip install tensorflow tensorflow-hub pandas numpy
```

Usage
1. Open the Jupyter Notebook: Open the Skimlit_ML_Assignment.ipynb notebook in Jupyter or a compatible IDE like Jupyter Lab or Google Colab.
2. Run Cells Sequentially: Follow the instructions in each cell.

The notebook will:
1. Load and preprocess data
2. Train a text classification model
3. Evaluate the model on validation data
4. Display results in a structured format

## Saving and Loading the Model
To save the trained model:

```
model.save("skimlit_model.h5")
```

To load and use the saved model:

```
from tensorflow.keras.models import load_model
model = load_model("skimlit_model.h5")
```

## Evaluating Model Performance
Run the final cells in the notebook to evaluate the model on a validation dataset.

The model is evaluated based on prediction accuracy across different abstract sections. Metrics such as accuracy and F1 score are calculated and printed to assess performance.

## Results
The code includes a sample output predicted using the best performing model out of all the tested models.
