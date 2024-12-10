# Machine Learning

## Developers
* (ML) Fanny Rorencia Ribowo - Universitas Surabaya
* (ML) Janet Deby Marlien Manoach - Universitas Surabaya
* (ML) Theophilus - Institut Informatika Indonesia Surabaya


## Table Of Content
- [Developers](#developers)
- [Directory](#directory)
- [Project Structure](#project-structure)
- [Installation](#installation)
    - [1. Clone The Repositories](#1-clone-the-repositories)
    - [2. Setup Environment](#2-setup-environment)
- [Use the Virtual Environment in VS Code](#use-the-virtual-environment-in-vs-code)
- [Using Pre-trained Models](#using-pre-trained-models)
- [Train the Models](#train-the-models-optional) (Optional)
- [Evaluate the Models](#evaluate-the-models)
- [Prediction](#prediction)
- [Links](#links)
    - [GitHub](#github)
    - [LinkedIn](#linkedin)

## Directory
<pre>
│  README.md
│  requirements.txt
│  modelBMI_v1.ipynb
│  model_v3_acc_98.04_loss_0.33.keras
│  model_v3.ipynb
│  bmi_model-accuracy_0.90-loss_0.26-val_accuracy_0.91-val_loss_0.23.keras
│  .gitignore
│
├─saved_models_v3_image_2
│  ├─weights-improvment-01-2.50.keras
│  ├─weights-improvment-02-1.66.keras
│  ├─...
│  └─weights-improvment-15-0.33.keras
│
├─processed_dataset
│  ├─wortel
│  ├─udang
│  ├─...
│  └─ayam
│     ├─ayam0.jpg
│     └─...
│
├─dataset_split
│  ├─wortel
│  ├─udang
│  ├─...
│  └─ayam
│     ├─ayam0.jpg
│     └─...
│
├─dataset
│  ├─test
│  ├─train
│  └─validation
│     ├─wortel
│     ├─udang
│     ├─...
│     └─ayam
│        ├─ayam0.jpg
│        └─...
│
└─csv
    ├─bmi_validation.csv
    ├─bmi_train.csv
    └─bmi.csv
</pre>
## Project Structure
- **`modelBMI_v1.ipynb`**: Jupyter Notebook for training or using the BMI model.
- **`model_v3.ipynb`**: Jupyter Notebook for training or using the image classification model.
- **`model_v3_acc_98.04_loss_0.33.keras`**: Pre-trained model for image classification.
- **`bmi_model-accuracy_0.90-loss_0.26-val_accuracy_0.91-val_loss_0.23.keras`**: Pre-trained BMI prediction model.
- **`saved_models_v3_image_2/`**: Checkpoints for the image classification model.
- **`processed_dataset/`**: Preprocessed datasets grouped by class (e.g., wortel, udang).
- **`dataset/`**: Original dataset with train, test, and validation splits that through image scraping from Google Images.
- **`csv/`**: CSV files containing BMI-related data for training and validation from kaggle.

## Installation
### 1. Clone The Repositories
#### Clone using the web URL.
```
git clone https://github.com/NYAM-Food-App/Machine-Learning.git
```
#### Using GitHub official CLI.
```
gh repo clone NYAM-Food-App/Machine-Learning
```
**Recommended IDE:** Visual Studio Code
### 2. Setup Environment
#### Setup Environment - Anaconda
```
conda create --name nyam-ml python=3.12.7
conda activate nyam-ml
pip install -r requirements.txt
```

#### Setup Environment - Shell/Terminal
```
mkdir Machine-Learning
cd Machine-Learning
pipenv install
pipenv shell
pip install -r requirements.txt
```

## Use the Virtual Environment in VS Code

If you want to run the project in **VS Code**, make sure you configure VS Code to use the virtual environment you created. Follow these steps:

#### 1. **Select the Virtual Environment in VS Code**
1. Open the project folder in VS Code.
2. Press **Ctrl + Shift + P** (Windows/Linux) or **Cmd + Shift + P** (Mac) to open the Command Palette.
3. Type and select **Python: Select Interpreter**.
4. From the list, select the interpreter that corresponds to your virtual environment (e.g., `nyam-ml`).  
   It should look something like this:
   ```
   (base) .../anaconda3/envs/nyam-ml/bin/python
   ```

#### 2. **Run Notebooks in the Virtual Environment**
1. Open a Jupyter Notebook file (e.g., `modelBMI_v1.ipynb` or `model_v3.ipynb`).
2. At the top right of the notebook interface, click on the kernel name (or "Select Kernel").
3. Choose the interpreter that corresponds to your virtual environment.

3. If you haven't created the environment yet, follow the steps in the [**Setup Environment**](#2-setup-environment) section above to create and activate it.

## Using Pre-trained Models

1. **For BMI Prediction:**
   - Open the `modelBMI_v1.ipynb` notebook.
   - Load the pre-trained model:
     ```python
     from tensorflow.keras.models import load_model
     model = load_model('bmi_model-accuracy_0.90-loss_0.26-val_accuracy_0.91-val_loss_0.23.keras')
     ```
   - Follow the notebook instructions to test the model with your BMI data in `csv/bmi.csv`.

2. **For Image Classification:**
   - Open the `model_v3.ipynb` notebook.
   - Load the pre-trained model:
     ```python
     from tensorflow.keras.models import load_model
     model = load_model('model_v3_acc_98.04_loss_0.33.keras')
     ```
   - Follow the notebook instructions to classify images in the `dataset/test/` directory.

## Train the Models (Optional)

1. **BMI Model:**
   - Use the `modelBMI_v1.ipynb` notebook.
   - Update the `csv/bmi_train.csv` and `csv/bmi_validation.csv` files if you want to use new data.

2. **Image Classification Model:**
   - Use the `model_v3.ipynb` notebook.
   - Update the `dataset/train/` and `dataset/validation/` directories with new data.
   - Modify the training parameters as needed.

## Evaluate the Models

1. **Run the evaluation cells in the notebooks** (`modelBMI_v1.ipynb` or `model_v3.ipynb`) to see the models' performance on the test datasets.

## Prediction

1. **For BMI:**
   - Run the prediction cells in `modelBMI_v1.ipynb`.

2. **For Images:**
   - Run the prediction cells in `model_v3.ipynb`.

## Links
#### GitHub
* [GitHub - NYAM-Food-App](https://github.com/NYAM-Food-App)
* [GitHub - Machine Learning](https://github.com/NYAM-Food-App/Machine-Learning)
#### LinkedIn
* [Janet Deby Marlien Manoach](https://www.linkedin.com/in/deby-manoach/)
* [Fanny Rorencia Ribowo](https://www.linkedin.com/in/fanny-rorencia-ribowo-27390b228/)
* [Theophilus](https://www.linkedin.com/in/theophilus-a3567a331/)
#### Kaggle
* [BMI - Body Mass Index](https://www.kaggle.com/datasets/sjagkoo7/bmi-body-mass-index)
* [BMI Dataset](https://www.kaggle.com/datasets/yasserh/bmidataset?resource=download)