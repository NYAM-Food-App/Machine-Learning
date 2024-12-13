<div align="center">
  <h1>NYAM 
  
  (Not Your Average Menu)

  🤖Machine Learning🤖 
  
  Repository </h1>
  <p>Two powerful machine learning models for image classification and BMI prediction, built with TensorFlow!</p>
</div>

---

## 📌 Summary
We developed two machine learning models using TensorFlow for our application:

1. **Image Classification Model:** Utilizes the MobileNetV2 architecture to identify ingredients from images, categorized into 17 classes.
2. **BMI Prediction Model:** Leverages Body Mass Index (BMI) data to classify individuals into six distinct BMI categories.

Data preprocessing steps included:
- Resizing, sharpening, and categorizing image data.
- Cleaning the BMI dataset through missing value imputation and duplicate removal.

Both models are saved in `.keras` format for deployment and predictions on new data.

---
# 📚 Developers
![Developers Profile](assets/developers.png)

---
# 📖 Table Of Content
- [📌 Summary](#📌-summary)
- [📚 Developers](#📚-developers)
- [📂 Directory](#📂-directory)
- [🏗️ Project Structure](#️🏗️-project-structure)
- [💻 Installation](#💻-installation)
- [🖥️ Use the Virtual Environment in VS Code](#️🖥️-use-the-virtual-environment-in-vs-code)
- [📥 Input and Output Description](#📥-input-and-output-description)
- [🔄 General Workflow](#🔄-general-workflow)
- [📦 Using Pre-trained Models](#📦-using-pre-trained-models)
- [🏋️ Train the Models](#🏋️-train-the-models-optional) (Optional)
- [🔗 Links](#🔗-links)
---
# 📂 Directory
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
├─csv
│  ├─bmi_validation.csv
│  ├─bmi_train.csv
│  └─bmi.csv
│
└─assets
   ├─developers.png
   └─...
</pre>
---

## 🏗️ Project Structure
| File/Directory                      | Description                                                                                     |
|-------------------------------------|-------------------------------------------------------------------------------------------------|
| `modelBMI_v1.ipynb`                 | Jupyter Notebook for training or using the BMI model.                                          |
| `model_v3.ipynb`                    | Jupyter Notebook for training or using the image classification model.                         |
| `model_v3_acc_98.04_loss_0.33.keras`| Pre-trained model for image classification.                                                    |
| `bmi_model-accuracy_0.90-loss_0.26.keras` | Pre-trained BMI prediction model.                                                        |
| `saved_models_v3_image_2/`          | Checkpoints for the image classification model.                                                |
| `processed_dataset/`                | Preprocessed datasets grouped by class (e.g., wortel, udang).                                  |
| `dataset_split/`                    | Splitted dataset for training, testing, and validation.                                        |
| `dataset/`                          | Original dataset collected via image scraping.                                                 |
| `csv/`                              | BMI-related CSV files for training and validation.                                             |
| `assets/`                           | Image and other asset files.                                                                   |

---
# 💻 Installation
### 1. Clone The Repositories
#### Clone using the web URL.
```
git clone https://github.com/NYAM-Food-App/Machine-Learning.git
```
#### Using GitHub official CLI.
```
gh repo clone NYAM-Food-App/Machine-Learning
```
**Recommended IDE**: [Visual Studio Code](https://code.visualstudio.com/)
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
---
# 🖥️ Use the Virtual Environment in VS Code

If you want to run the project in **VS Code**, make sure you configure VS Code to use the virtual environment you created. Follow these steps:

### 1. **Select the Virtual Environment in VS Code**
1. Open the project folder in VS Code.
2. Press **Ctrl + Shift + P** (Windows/Linux) or **Cmd + Shift + P** (Mac) to open the Command Palette.
3. Type and select **Python: Select Interpreter**.
4. From the list, select the interpreter that corresponds to your virtual environment (e.g., `nyam-ml`).  
   It should look something like this:
   ```
   (base) .../anaconda3/envs/nyam-ml/bin/python
   ```
---

### 2. **Run Notebooks in the Virtual Environment**
1. Open a Jupyter Notebook file (e.g., `modelBMI_v1.ipynb` or `model_v3.ipynb`).
2. At the top right of the notebook interface, click on the kernel name (or "Select Kernel").
3. Choose the interpreter that corresponds to your virtual environment.

3. If you haven't created the environment yet, follow the steps in the [**Setup Environment**](#2-setup-environment) section above to create and activate it.
---
# 📥 Input and Output Description

### 1. **BMI Prediction Model**
- **Input**:  
  A CSV file containing the following columns:
  - `gender`: (e.g., "0" for Male or "1" for female)
  - `height`: (in cm)
  - `weight`: (in kg)
  - `BMI`: (weight) / (height ^ 2) * 10,000

  
  
  Example:
  ```csv
  gender,   height,  weight,  BMI
  0,        170,     70,      24.22
  1,        160,     55,      21.48
  ```

- **Output**:  
  There are 6 label BMI value for each row from 0 to 5.
   - 0: Extremely Weak  
   - 1: Weak  
   - 2: Normal  
   - 3: Overweight  
   - 4: Obesity  
   - 5: Extreme Obesity  

---

### 2. **Image Classification Model**
- **Input**:  
  Image files that will be preproccessed like resized it to 224 x 224 and more (e.g., `ayam0.jpg`, `wortel1.jpg`).
  
- **Output**:  
  - There are 17 predicted ingredient foods class (e.g., "Chicken", "Carrot", "Shrimp").
  - The model's confidence score from 0 to 1 for each class (e.g., `[Ayam: 0.98, Wortel: 0.01, Udang: 0.01]`).

---

# 🔄 General Workflow
### **BMI Prediction:**
1. Prepare a CSV file with data (e.g., `csv/bmi.csv`).
2. Run `modelBMI_v1.ipynb`.
3. Load and predict BMI categories.

### **Image Classification:**
1. Prepare the test image.
2. Run `model_v3.ipynb`.
3. Predict the uploaded ingredient food images.

---


# 📦 Using Pre-trained Models

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
   - Follow the notebook instructions to classify images in the `dataset/` directory.
---
# 🏋️ Train the Models (Optional)

1. **BMI Model:**
   - Use the `modelBMI_v1.ipynb` notebook.
   - Update the `csv/bmi_train.csv` and `csv/bmi_validation.csv` files if you want to use new data.

2. **Image Classification Model:**
   - Use the `model_v3.ipynb` notebook.
   - Update the `dataset_split/train/`, `dataset_split/validation/` and `dataset_split/test/` directories with new data by running the preprocessing code.
   - Modify the training parameters as needed.

---

# 🔗 Links
### ![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat-square&logo=github&logoColor=white)
- [NYAM-Food-App](https://github.com/NYAM-Food-App)
- [Machine Learning](https://github.com/NYAM-Food-App/Machine-Learning)
### ![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5.svg?style=flat-square&logo=linkedin&logoColor=white)
- [Janet Deby Marlien Manoach](https://www.linkedin.com/in/deby-manoach/)
- [Fanny Rorencia Ribowo](https://www.linkedin.com/in/fanny-rorencia-ribowo-27390b228/)
- [Theophilus](https://www.linkedin.com/in/theophilus-a3567a331/)
### ![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat-square&logo=github&logoColor=white)
- [BMI - Body Mass Index](https://www.kaggle.com/datasets/sjagkoo7/bmi-body-mass-index)
- [BMI Dataset](https://www.kaggle.com/datasets/yasserh/bmidataset?resource=download)
---