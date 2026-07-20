# Disease Prediction Assistant

Predicts likely diseases from user-selected symptoms using classification
models (Logistic Regression, KNN, Decision Tree), with a Streamlit web
interface.

## Project structure

```
disease_predictor/
├── generate_dataset.py   # creates dataset.csv (symptom-disease training data)
├── train_model.py        # trains & compares 3 classifiers, saves the best one
├── app.py                 # Streamlit web app for interactive prediction
├── requirements.txt
├── dataset.csv            # generated data (created by generate_dataset.py)
├── model.pkl               # trained model (created by train_model.py)
├── label_encoder.pkl       # disease label encoder (created by train_model.py)
└── symptoms.pkl             # ordered list of symptom feature names
```

## How to run it

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate the training data**
   ```bash
   python generate_dataset.py
   ```
   This creates `dataset.csv` — a synthetic-but-medically-informed dataset
   mapping symptom combinations to 20 common diseases. (You can swap this
   for a real dataset later — see "Using a real dataset" below.)

3. **Train and compare the models**
   ```bash
   python train_model.py
   ```
   This trains Logistic Regression, KNN, and a Decision Tree, prints
   accuracy + a classification report for each, and saves the
   best-performing model as `model.pkl`.

4. **Launch the web app**
   ```bash
   streamlit run app.py
   ```
   This opens a browser tab where you can check off symptoms and get a
   predicted disease plus a top-5 probability ranking.

## How it works (pipeline)

1. **Data representation**: each row is a patient record. Every symptom is
   a binary column (1 = present, 0 = absent). The label column is the
   diagnosed disease.
2. **Preprocessing**: disease names are label-encoded into integers with
   `sklearn.preprocessing.LabelEncoder`.
3. **Train/test split**: 80/20 split, stratified by disease so every class
   is represented in both sets.
4. **Model training**: three classifiers are trained on the same data:
   - **Logistic Regression** — a strong linear baseline, usually the most
     robust of the three on this kind of data.
   - **K-Nearest Neighbors (KNN)** — predicts based on the most similar
     symptom patterns seen during training.
   - **Decision Tree** — learns simple if/then symptom rules; easiest to
     interpret but prone to overfitting on small data.
5. **Model selection**: whichever model scores highest on the held-out
   test set is saved and used by the app.
6. **Deployment**: `app.py` loads the saved model and gives users
   checkboxes for every symptom; on submit, it builds the same feature
   vector format used in training and calls `model.predict()` /
   `model.predict_proba()`.

## Using a real dataset instead

For a stronger, more credible project, replace the synthetic data with a
public dataset such as **"Disease Prediction Using Machine Learning"** on
Kaggle (132 symptom columns, 41 diseases, ~5000 rows). Steps:

1. Download `Training.csv` / `Testing.csv` from Kaggle.
2. Save as `dataset.csv` in this folder, keeping symptom columns + a
   `disease`/`prognosis` column as the label (rename the label column to
   `disease` or adjust `train_model.py`'s reference to it).
3. Re-run `python train_model.py` — no other code changes needed, since
   the training script reads column names dynamically.

## Ideas to extend the project

- Add more models (Random Forest, SVM, Naive Bayes) and compare all of them.
- Add symptom **search/autocomplete** instead of scrolling checkboxes.
- Show a **confusion matrix** and **feature importance** chart in the app.
- Add **severity weighting** (e.g. "high fever" contributes more than "mild fatigue").
- Deploy publicly via **Streamlit Community Cloud** (free, connects directly to a GitHub repo).
- Add a disclaimer + links to real medical resources for each predicted disease.

## Disclaimer

This project is for **educational purposes only**. It is not a diagnostic
tool and must not be used as a substitute for professional medical advice.
