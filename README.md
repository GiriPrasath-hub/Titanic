# Titanic Survival Prediction Classification System

This repository implements a production-ready Machine Learning Classification System using a **Decision Tree Classifier** to predict whether a passenger survived the Titanic disaster.

---

## 📋 Objective
To analyze the Titanic passengers' data, train an optimized Decision Tree Classifier, and build an interactive prediction dashboard using Streamlit to serve predictions in real-time.

---

## 🗄️ Dataset & Findings
The dataset is located in `dataset/train_and_test2.csv`.
- **Shape:** 1,309 rows, 28 columns.
- **Pre-cleaning Status:** The raw dataset was partially pre-cleaned. Numerical columns like `Sex` and `Embarked` are pre-encoded. Text columns like `Name`, `Ticket`, and `Cabin` are missing.
- **Redundant Columns:** Dropped 19 columns named `zero` through `zero.18` containing only `0` values.
- **Target Feature:** The target feature was named `2urvived` and has been normalized to `Survived`.
  - Class `0` (Did Not Survive): 967 rows (73.87%)
  - Class `1` (Survived): 342 rows (26.13%)
- **Missing Values:** Only `Embarked` contained 2 missing values, which were successfully imputed using mode imputation.

---

## 🌲 Algorithm & Model Details
We used a **DecisionTreeClassifier** tuned via **GridSearchCV** with 5-fold stratified cross-validation.
- **Optimal Hyperparameters:**
  - `criterion`: `'gini'`
  - `max_depth`: `3`
  - `min_samples_leaf`: `4`
  - `min_samples_split`: `2`
  - `max_features`: `None`
- **Performance:**
  - **Cross-Validation ROC-AUC:** `0.7993`
  - **Training Accuracy:** `81.18%`
  - **Testing Accuracy:** `75.19%`
  - **Testing Precision:** `53.19%`
  - **Testing Recall:** `36.76%`
  - **Weighted F1:** `0.7356`

---

## 📁 Folder Structure
```text
Titanic/
├── dataset/
│   └── train_and_test2.csv     # Raw dataset
├── model/
│   ├── encoder.pkl             # Fitted OneHotEncoder
│   ├── feature_columns.pkl     # Preprocessed feature columns list
│   ├── model.pkl               # Complete fitted sklearn Pipeline
│   └── preprocessor.pkl        # Fitted ColumnTransformer
├── notebooks/
│   └── EDA.ipynb               # Exploratory Data Analysis notebook
├── utils/
│   ├── __init__.py             # Package init imports
│   ├── predict.py              # Prediction engine module
│   └── preprocess.py           # Preprocessing and cleaning functions
├── app.py                      # Streamlit Application
├── .gitignore                  # Git ignore rules
├── README.md                   # Project documentation
├── requirements.txt            # Package dependencies
└── train_model.py              # Model training and serialization script
```

---

## ⚙️ Workflow Diagram (ASCII)
```text
       +---------------------------------------------+
       |          Raw Dataset loading                |
       |      (dataset/train_and_test2.csv)          |
       +----------------------+----------------------+
                              |
                              v
       +----------------------+----------------------+
       |       normalize_column_names()              |
       |  - Standardize target '2urvived' ->'Survived'|
       |  - Clean leading spaces/numbers/symbols     |
       +----------------------+----------------------+
                              |
                              v
       +----------------------+----------------------+
       |           clean_dataset()                   |
       |  - Drop 19 redundant 'zero' columns         |
       |  - Drop Passengerid                         |
       +----------------------+----------------------+
                              |
                              v
       +----------------------+----------------------+
       |          engineer_features()                |
       |  - Create FamilySize (sibsp + Parch + 1)   |
       |  - Create IsAlone (FamilySize == 1)         |
       +----------------------+----------------------+
                              |
                              v
       +----------------------+----------------------+
       |      Pipeline & ColumnTransformer           |
       |  - Median Imputation for numerical features |
       |  - Mode Imputation for categories           |
       |  - OneHotEncoder on Pclass and Embarked     |
       +----------------------+----------------------+
                              |
                              v
       +----------------------+----------------------+
       |      GridSearchCV & Model Fitting           |
       |  - Train Decision Tree Classifier           |
       |  - Export components to model/*.pkl         |
       +---------------------------------------------+
```

---

## 🚀 Installation & Setup
1. Clone or navigate to the project directory:
   ```bash
   cd c:\AIML\Interview\Titanic
   ```
2. Set up a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## ⚙️ How to Run

### 1. Training the Model
To retrain the model and save the serialized pickle artifacts, run:
```bash
python train_model.py
```

### 2. Launching the Web App
To start the Streamlit web dashboard locally:
```bash
streamlit run app.py
```

---

## 🌐 Deployment
This application is fully compatible with **Streamlit Cloud** and **GitHub**:
- Resolves all paths using dynamic, platform-independent relative paths.
- Requires no absolute local references.
- Configured with locked package versions in `requirements.txt`.

---

## 🖼️ Screenshots Placeholder
*(Add dashboard interface screenshots here)*

---

## 🔮 Future Improvements
- **Alternative Algorithms:** Evaluate Random Forests and Gradient Boosted Trees to improve Recall and ROC-AUC.
- **Extended Preprocessing:** Explore custom scaling for continuous variables (though not required by trees, it assists comparative models).
- **Synthetic Balancing:** Implement SMOTE to handle the target variable class imbalance.
