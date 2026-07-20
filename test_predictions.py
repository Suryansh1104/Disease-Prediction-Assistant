"""
test_predictions.py
--------------------
Manually check whether the trained model gives sensible predictions,
by feeding it known symptom combinations and seeing whether it agrees.

Run:
    python test_predictions.py
"""

import pickle
import pandas as pd

with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)
with open("symptoms.pkl", "rb") as f:
    symptom_cols = pickle.load(f)

# ---------------------------------------------------------------------
# Add / edit test cases here. Each is (symptoms_you_pick, disease_you_expect)
# ---------------------------------------------------------------------
test_cases = [
    (["sneezing", "runny_nose"], "Common Cold"),
    (["high_fever", "dry_cough", "loss_of_smell"], "COVID-19"),
    (["itchy_rash", "red_spots"], "Chickenpox"),
    (["burning_urination", "frequent_urination"], "Urinary Tract Infection"),
    (["headache", "nausea", "sensitivity_to_light"], "Migraine"),
    (["high_fever", "chills", "sweating"], "Malaria"),
    (["joint_pain", "joint_stiffness"], "Arthritis"),
    (["frequent_urination", "excessive_thirst", "weight_loss"], "Diabetes"),
    (["heartburn", "regurgitation"], "Acid Reflux (GERD)"),
    (["wheezing", "shortness_of_breath", "chest_tightness"], "Asthma"),
]

correct = 0
for symptoms, expected in test_cases:
    vec = pd.DataFrame(
        [[1 if s in symptoms else 0 for s in symptom_cols]],
        columns=symptom_cols,
    )
    pred = model.predict(vec)[0]
    result = le.inverse_transform([pred])[0]

    is_correct = result == expected
    correct += is_correct
    mark = "PASS" if is_correct else "FAIL"

    print(f"[{mark}] symptoms={symptoms}")
    print(f"       predicted: {result}   expected: {expected}")

    # If it got it wrong, show what it was actually considering
    if not is_correct and hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)[0]
        top3 = sorted(zip(le.classes_, proba), key=lambda x: -x[1])[:3]
        print(f"       top 3 guesses: {[(d, round(p,2)) for d, p in top3]}")
    print()

print(f"Result: {correct}/{len(test_cases)} test cases correct")
