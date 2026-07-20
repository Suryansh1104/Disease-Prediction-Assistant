"""
train_model.py
---------------
Loads dataset.csv, trains three classifiers (Logistic Regression, KNN,
Decision Tree), compares them on a held-out test set, and saves:
    - the best-performing model  -> model.pkl
    - the fitted label encoder   -> label_encoder.pkl
    - the list of symptom columns (feature order) -> symptoms.pkl

Run:
    python train_model.py
"""

import json
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


def main():
    # 1. Load data --------------------------------------------------
    df = pd.read_csv("dataset.csv")
    symptom_cols = [c for c in df.columns if c != "disease"]

    X = df[symptom_cols]
    y_raw = df["disease"]

    # 2. Encode disease labels as integers ---------------------------
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # 3. Train/test split --------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Define candidate models ---------------------------------------
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        # 'jaccard' suits sparse binary symptom vectors far better than the
        # default Euclidean distance -- it compares symptoms in common
        # relative to symptoms present at all, instead of being dominated
        # by how many total symptoms each patient happens to have checked.
        "KNN": KNeighborsClassifier(n_neighbors=5, metric="jaccard"),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    }

    results = {}
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    best_name, best_model, best_acc = None, None, -1

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # 5-fold cross-validation for a more robust estimate
        cv_scores = cross_val_score(model, X, y, cv=5)

        results[name] = acc
        print(f"\n{name}")
        print(f"  Test accuracy      : {acc:.4f}")
        print(f"  CV accuracy (5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        if acc > best_acc:
            best_name, best_model, best_acc = name, model, acc

    print("\n" + "=" * 60)
    print(f"BEST MODEL: {best_name} (test accuracy = {best_acc:.4f})")
    print("=" * 60)

    # Detailed report for the winning model
    preds = best_model.predict(X_test)
    print("\nClassification report for best model:\n")
    print(classification_report(y_test, preds, target_names=le.classes_, zero_division=0))

    # 5. Persist model + encoder + feature order ------------------------
    with open("model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    with open("symptoms.pkl", "wb") as f:
        pickle.dump(symptom_cols, f)

    # Save metrics so the Streamlit app can display real numbers instead
    # of hardcoded ones
    metrics = {
        "best_model_name": best_name,
        "best_model_accuracy": round(best_acc, 4),
        "all_results": {name: round(acc, 4) for name, acc in results.items()},
        "n_diseases": len(le.classes_),
        "n_symptoms": len(symptom_cols),
        "n_training_rows": len(df),
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved '{best_name}' -> model.pkl")
    print("Saved label_encoder.pkl, symptoms.pkl, and metrics.json")


if __name__ == "__main__":
    main()
