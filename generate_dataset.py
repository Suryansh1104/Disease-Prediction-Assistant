"""
generate_dataset.py
--------------------
Generates a synthetic (but medically-informed) symptom -> disease dataset
for training the classifiers.

In a real-world version of this project you would instead download a
public dataset such as the "Disease Prediction Using Machine Learning"
dataset on Kaggle (132 symptom columns x 41 diseases) and skip this
step entirely. This script exists so the project is fully runnable
offline, with no external downloads required.

Output: dataset.csv
    Columns: symptom_1 ... symptom_N (0/1), disease (label)
"""

import random
import pandas as pd

random.seed(42)

# ---------------------------------------------------------------------
# 1. Define diseases and the symptoms typically associated with each.
#    (Simplified for teaching purposes -- not medical advice.)
# ---------------------------------------------------------------------
DISEASE_SYMPTOMS = {
    "Common Cold": ["sneezing", "runny_nose", "sore_throat", "cough", "mild_fever", "congestion"],
    "Flu": ["high_fever", "body_ache", "fatigue", "cough", "headache", "chills", "sore_throat"],
    "COVID-19": ["high_fever", "dry_cough", "loss_of_smell", "fatigue", "shortness_of_breath", "sore_throat"],
    "Migraine": ["headache", "nausea", "sensitivity_to_light", "sensitivity_to_sound", "vomiting"],
    "Chickenpox": ["itchy_rash", "mild_fever", "fatigue", "loss_of_appetite", "red_spots"],
    "Malaria": ["high_fever", "chills", "sweating", "headache", "nausea", "muscle_pain"],
    "Typhoid": ["high_fever", "abdominal_pain", "weakness", "headache", "loss_of_appetite", "constipation"],
    "Dengue": ["high_fever", "severe_headache", "joint_pain", "muscle_pain", "rash", "nausea"],
    "Gastroenteritis": ["diarrhea", "vomiting", "abdominal_pain", "nausea", "mild_fever", "dehydration"],
    "Urinary Tract Infection": ["burning_urination", "frequent_urination", "abdominal_pain", "cloudy_urine", "mild_fever"],
    "Asthma": ["shortness_of_breath", "wheezing", "chest_tightness", "cough", "fatigue"],
    "Bronchitis": ["cough", "chest_discomfort", "fatigue", "mild_fever", "shortness_of_breath", "mucus"],
    "Pneumonia": ["high_fever", "cough", "chest_pain", "shortness_of_breath", "fatigue", "chills"],
    "Hypertension": ["headache", "dizziness", "blurred_vision", "chest_pain", "fatigue", "nosebleeds"],
    "Diabetes": ["frequent_urination", "excessive_thirst", "fatigue", "blurred_vision", "weight_loss"],
    "Anemia": ["fatigue", "pale_skin", "dizziness", "shortness_of_breath", "weakness"],
    "Allergic Rhinitis": ["sneezing", "runny_nose", "itchy_eyes", "congestion", "watery_eyes"],
    "Conjunctivitis": ["itchy_eyes", "watery_eyes", "redness_eyes", "discharge_eyes"],
    "Acid Reflux (GERD)": ["heartburn", "chest_discomfort", "regurgitation", "sore_throat", "nausea"],
    "Arthritis": ["joint_pain", "joint_stiffness", "swelling", "fatigue", "reduced_range_of_motion"],
    "Thyroid Disorder (Hypothyroidism)": ["fatigue", "weight_gain", "cold_intolerance", "dry_skin", "hair_loss", "constipation"],
    "Thyroid Disorder (Hyperthyroidism)": ["weight_loss", "heat_intolerance", "rapid_heartbeat", "tremor", "increased_appetite", "sweating"],
}

ALL_SYMPTOMS = sorted({s for symptoms in DISEASE_SYMPTOMS.values() for s in symptoms})

# A "hallmark" symptom is the single most distinguishing sign for a disease
# -- the one a doctor would specifically ask about to break a tie between
# look-alike conditions. When defined, it's forced to appear in most
# synthetic patients for that disease (see generate_record), so the model
# reliably learns to lean on it. This directly targets the disease pairs
# that get confused most often (identified via confusion matrix analysis):
# Anemia/Hypertension share fatigue+dizziness, and Allergic Rhinitis/
# Common Cold share sneezing+runny_nose+congestion almost entirely.
HALLMARK_SYMPTOMS = {
    "Anemia": "pale_skin",
    "Hypertension": "nosebleeds",
    "Allergic Rhinitis": "itchy_eyes",
    "Common Cold": "sore_throat",
    "Pneumonia": "chest_pain",
}

# ---------------------------------------------------------------------
# 2. Generate synthetic patient records.
#    Each record = a disease's "core" symptoms with some noise:
#      - a symptom may randomly be missing (not every patient has every symptom)
#      - a random unrelated symptom may occasionally appear (real-world noise)
# ---------------------------------------------------------------------
def generate_record(disease, min_fraction, max_fraction):
    """
    min_fraction/max_fraction control what proportion of a disease's core
    symptom list gets checked for this synthetic patient. Real users often
    only tick 1-3 boxes, so we deliberately train on sparse selections too
    (not just "rich" patients with most symptoms present) -- otherwise the
    model never learns what a minimal, realistic query looks like.
    """
    core = DISEASE_SYMPTOMS[disease]
    record = {s: 0 for s in ALL_SYMPTOMS}

    lo = max(1, int(len(core) * min_fraction))
    hi = max(lo, int(len(core) * max_fraction))
    n_present = random.randint(lo, hi)
    present = set(random.sample(core, n_present))

    # Force the disease's hallmark symptom to be present most of the time,
    # even if random sampling above missed it -- this is what lets the
    # model reliably distinguish diseases that otherwise share most of
    # their symptoms (e.g. Anemia vs Hypertension both have fatigue +
    # dizziness; pale_skin is what actually tells them apart).
    hallmark = HALLMARK_SYMPTOMS.get(disease)
    if hallmark and random.random() < 0.85:
        present.add(hallmark)

    for s in present:
        record[s] = 1

    # Small chance of an unrelated symptom (noise), makes the problem realistic
    if random.random() < 0.10:
        noisy_symptom = random.choice(ALL_SYMPTOMS)
        record[noisy_symptom] = 1

    record["disease"] = disease
    return record


def build_dataset(samples_per_disease=120):
    """
    For each disease, generate patients across a spread of symptom-set
    sizes. Skewed toward moderate/rich symptom presentation (3+ symptoms
    typical) rather than single, highly ambiguous symptoms -- this trades
    away some robustness on extremely sparse 1-symptom queries in exchange
    for meaningfully higher overall accuracy, since a single isolated
    symptom is often genuinely indistinguishable between several diseases.
    """
    # (min_fraction, max_fraction, share of samples_per_disease)
    profiles = [
        (0.45, 0.60, 0.35),  # moderate-sparse: ~3 symptoms typically
        (0.55, 0.80, 0.35),  # moderate
        (0.75, 1.00, 0.30),  # rich / textbook
    ]

    rows = []
    for disease in DISEASE_SYMPTOMS:
        for min_f, max_f, share in profiles:
            n = max(1, round(samples_per_disease * share))
            for _ in range(n):
                rows.append(generate_record(disease, min_f, max_f))

    df = pd.DataFrame(rows)
    cols = ALL_SYMPTOMS + ["disease"]
    return df[cols]


if __name__ == "__main__":
    df = build_dataset(samples_per_disease=60)
    df.to_csv("dataset.csv", index=False)
    print(f"Generated dataset.csv with {len(df)} rows and {len(ALL_SYMPTOMS)} symptom columns.")
    print(f"Diseases covered: {len(DISEASE_SYMPTOMS)}")
