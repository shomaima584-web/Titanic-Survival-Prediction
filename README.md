# Titanic Survival Prediction

A machine learning classification project that predicts whether a Titanic passenger survived, based on details such as ticket class, sex, age, family size and fare.

## Project Links
- **Live App:** https://titanic-survival-predictor-583.streamlit.app/
- **Demo Video:** https://drive.google.com/file/d/1xCyf5EE8f5ZE71qL6NcEUuYw5tz21QWV/view?usp=drive_link

## Dataset
Titanic dataset from Kaggle ("Titanic - Machine Learning from Disaster"), with 891 passengers and 12 columns.

## Workflow
1. Data loading and understanding
2. Exploratory Data Analysis (EDA)
3. Data cleaning (missing values in Age, Embarked and Cabin)
4. Feature engineering (FamilySize, IsAlone)
5. Encoding of categorical features
6. Training and comparing 4 models: Logistic Regression, Decision Tree, Random Forest, KNN
7. Evaluation (accuracy, classification report, confusion matrix, cross-validation)
8. Saving the best model and building a Streamlit web app

## Results
- Best model: K-Nearest Neighbors (KNN), tied with Random Forest
- Test accuracy: 81.56%

| Model | Accuracy |
|---|---|
| KNN | 81.56% |
| Random Forest | 81.56% |
| Logistic Regression | 80.45% |
| Decision Tree | 75.98% |

## Key Findings
- Females had a much higher survival rate than males.
- 1st class passengers had better chances of survival.
- Children and passengers who paid higher fares also had a higher survival rate.

## Repository Files
| File | Description |
|---|---|
| `Titanic_Survival_Prediction.ipynb` | Full notebook (EDA, modeling, evaluation) |
| `train.csv` | Dataset |
| `titanic_model.pkl` | Trained model |
| `app.py` | Streamlit web app |
| `requirements.txt` | Python dependencies |

## Run the App Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tech Stack
Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Streamlit
