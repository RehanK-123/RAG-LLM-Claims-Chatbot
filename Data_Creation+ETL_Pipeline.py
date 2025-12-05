import numpy as np 
import pandas as pd 
import os
import random
from faker import Faker
import numpy as np
# from sentence_transformers import SentenceTransformer
import faiss
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from dateutil import parser
from datetime import timedelta
import sqlite3
import re 
from transformers import AutoTokenizer, AutoModel   
import torch
import torch.nn.functional as F
from groq import Groq
import sqlparse

fake = Faker()

# ------------ CONFIGURATION ------------ #
NUM_PATIENTS = 1200
NUM_PROVIDERS = 180
NUM_CLAIMS = 4000

noise_probability = 0.12  # 12% rows receive noisy values
missing_probability = 0.007  # 7% fields randomly missing
typo_probability = 0.05  # chance of spelling errors


# ------------ REALISTIC MEDICAL DATA ------------ #
diagnosis_map = {
    "Diabetes Mellitus": ["Diabetes", "DM2", "Type 2 Diabetes", "T2D", "Diabtes"],
    "Hypertension": ["High BP", "HTN", "Hypertensoin"],
    "Asthma": ["Reactive airway disease"],
    "COVID-19": ["COVID", "SARS-CoV-2"],
    "Cancer": ["Carcinoma", "Oncology Case"],
    "Heart Disease": ["CAD", "Cardiac condition"]
}

icd_codes = {
    "Diabetes Mellitus": "E11.9",
    "Hypertension": "I10",
    "Asthma": "J45.909",
    "COVID-19": "U07.1",
    "Cancer": "C80.1",
    "Heart Disease": "I25.10"
}

specialties = [
    "Cardiology", "Endocrinology", "General Medicine",
    "Oncology", "Pulmonology", "Family Physician"
]

claim_status_distribution = {
    "Approved": 0.78,
    "Denied": 0.14,
    "Pending": 0.05,
    "Appealed": 0.03
}

denial_reasons = [
    "Missing documentation", "Incorrect coding",
    "Not covered service", "Eligibility issue",
    "Duplicate claim", "Policy expired"
]


# ------------ UTIL FUNCTIONS ------------ #

def maybe_missing(val):
    return random.choice([val, None, "", "N/A"]) if random.random() < missing_probability else val

def add_typo(text):
    if not isinstance(text, str) or len(text) < 4: return text
    if random.random() > typo_probability: return text
    idx = random.randint(0, len(text) - 2)
    return text[:idx] + text[idx+1] + text[idx] + text[idx+2:]


# ------------ GENERATE PATIENTS ------------ #
patients = []
for i in range(NUM_PATIENTS):
    age = int(np.clip(np.random.normal(48, 18), 1, 95))
    patients.append({
        "patient_id": f"P{i:05d}",
        "name": fake.name(),
        "age": age,
        "gender": random.choice(["Male", "Female", "Other"]),
        "city": fake.city()
    })

patients_df = pd.DataFrame(patients)


# ------------ GENERATE PROVIDERS ------------ #
providers = []
for i in range(NUM_PROVIDERS):
    providers.append({
        "provider_id": f"D{i:04d}",
        "name": fake.name(),
        "specialty": random.choice(specialties),
    })

providers_df = pd.DataFrame(providers)


# ------------ GENERATE CLAIMS ------------ #
claims = []

for i in range(NUM_CLAIMS):
    patient = random.choice(patients)
    provider = random.choice(providers)

    diag_key = random.choice(list(diagnosis_map.keys()))
    diagnosis = random.choice(diagnosis_map[diag_key])

    claim_amount = round(np.random.lognormal(mean=8, sigma=0.4), 2)  # Realistic skew
    service_date = fake.date_between(start_date="-3y", end_date="today")
    submission_date = service_date + timedelta(days=random.randint(2, 21))
    decision_date = submission_date + timedelta(days=random.randint(7, 120))

    status = random.choices(
        list(claim_status_distribution.keys()),
        list(claim_status_distribution.values())
    )[0]

    denial_reason = random.choice(denial_reasons) if status == "Denied" else "Passed"

    record = {
        "claim_id": f"CLM{i:06d}",
        "patient_id": patient["patient_id"],
        "provider_id": provider["provider_id"],
        "diagnosis": diagnosis,
        "icd10_code": icd_codes[diag_key],
        "claim_amount": claim_amount,
        "service_date": service_date,
        "submission_date": submission_date,
        "decision_date": decision_date,
        "status": status,
        "denial_reason": denial_reason,
        "notes": random.choice([
            "", "Resubmitted with corrected documents", "Urgent review requested",
            "Appeal pending", "Manual review"
        ])
    }

    # Apply random missing fields
    record = {k: maybe_missing(v) for k, v in record.items()}

    # Add typos intentionally
    record["diagnosis"] = add_typo(record["diagnosis"])
    record["notes"] = add_typo(record["notes"])

    claims.append(record)

claims_df = pd.DataFrame(claims)


# ------------ SAVE FILES ------------ #
patients_df.to_csv("patients.csv", index=False)
providers_df.to_csv("providers.csv", index=False)
claims_df.to_csv("claims.csv", index=False)

print("Mock dataset generated with noise, real-world variance, and saved as CSVs!")

#view Datasets

claims_df = pd.read_csv("C:\\Users\\ASUS\\Downloads\\Abacus Insights - RAG + LLM\\Abacus Insights - RAG + LLM\\claims.csv")
patients_df = pd.read_csv("C:\\Users\\ASUS\\Downloads\\Abacus Insights - RAG + LLM\\Abacus Insights - RAG + LLM\\patients.csv")
providers_df = pd.read_csv("C:\\Users\\ASUS\\Downloads\\Abacus Insights - RAG + LLM\\Abacus Insights - RAG + LLM\\providers.csv")

print(claims_df.head(), providers_df.head(), patients_df)

# -----------------------
#ETL pipeline


#------EXTRACT------#

claims_df = pd.read_csv("C:\\Users\\ASUS\\Downloads\\Abacus Insights - RAG + LLM\\Abacus Insights - RAG + LLM\\claims.csv")
patients_df = pd.read_csv("C:\\Users\\ASUS\\Downloads\\Abacus Insights - RAG + LLM\\Abacus Insights - RAG + LLM\\patients.csv")
providers_df = pd.read_csv("C:\\Users\\ASUS\\Downloads\\Abacus Insights - RAG + LLM\\Abacus Insights - RAG + LLM\\providers.csv")

#-------TRANSFORM------#
"""We drop the rows that contain important information that if null would cause the real world insurance payer to return the claim."""

# Normalize “fake nulls”
cleanup_values = ["", " ", "N/A", "n/a", "NA", "None", "null", "NULL"]
claims_df.replace(cleanup_values, np.nan, inplace=True)

# Define critical fields
critical_columns = [
    "claim_id",
    "patient_id",
    "provider_id",
    "diagnosis",
    "claim_amount",
    "service_date",
    "submission_date",
    "decision_date",
    "status",
]

# Denial reason only required if status == 'Denied'
def reason_missing(row):
    if row["status"] == "Denied" and pd.isna(row["denial_reason"]):
        return True
    return False


# Categorize validation issues like real payer systems
def categorize_issue(row):
    if pd.isna(row["claim_id"]): return "Missing Claim ID"
    if pd.isna(row["patient_id"]): return "Missing Patient ID"
    if pd.isna(row["provider_id"]): return "Missing Provider ID"
    if pd.isna(row["diagnosis"]): return "Missing Diagnosis / ICD Code"
    if pd.isna(row["claim_amount"]): return "Missing Billed Amount"
    if pd.isna(row["service_date"]): return "Missing Service Date"
    if pd.isna(row["submission_date"]): return "Missing Submission Date"
    if pd.isna(row["decision_date"]): return "Missing Decision Date"
    if pd.isna(row["icd10_code"]): return "Missing IDC10 code"
    if reason_missing(row): return "Denied Claim Missing Denial Reason"
    return None


# Apply classification
claims_df["review_reason"] = claims_df.apply(categorize_issue, axis=1)

# Split into two datasets
review_queue = claims_df[claims_df["review_reason"].notna()].copy()
claims_df = claims_df[claims_df["review_reason"].isna()].drop("review_reason", axis= 1)

# print(claims_df.head(), review_queue)
claims_df = claims_df.reset_index(drop= True)

#-------Impute the denial_reason & notes field---------#

# model = SentenceTransformer("all-MiniLM-L6-v2")
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def embed_text(text):
    # Tokenize input
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

    # Forward pass
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Mean Pooling (same as SentenceTransformer)
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize (optional but recommended)
    embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element contains token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def serialize(row):
    return (
        f"Diagnosis: {row['diagnosis']}. "
        f"Amount: {row['claim_amount']}. "
        f"Status: {row['status']}. "
        f"Denial reason: {row['denial_reason']}. "
        f"Notes: {row['notes']}."
    )

claims_df["text_embedding_input"] = claims_df.apply(serialize, axis=1)

embeddings = embed_text(claims_df["text_embedding_input"].tolist()).cpu().numpy()

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # cosine similarity (normalized vectors)
index.add(embeddings)

print(f"Indexed {index.ntotal} records.")
K = 5

def impute_with_knn(row, idx):
    query_vec = embeddings[idx].reshape(1, -1)
    distances, neighbors = index.search(query_vec, K+1)

    neighbors = neighbors[0][1:]
    neighbor_rows = claims_df.iloc[neighbors]
    
    denial_reason = row["denial_reason"]
    if pd.isna(denial_reason):
        candidates = neighbor_rows["denial_reason"].dropna().tolist()
        if candidates:
            denial_reason = max(set(candidates), key=candidates.count)
        else:
            denial_reason = "Not Applicable"

    # ---- Impute notes ----
    notes = row["notes"]
    if pd.isna(notes):
        candidates = neighbor_rows["notes"].dropna().tolist()
        if candidates:
            notes = candidates[0]
        else:
            notes = "No additional information provided."

    return denial_reason, notes


for idx, row in claims_df.iterrows():
    denial, note = impute_with_knn(row, idx)
    claims_df.at[idx, "denial_reason"] = denial
    claims_df.at[idx, "notes"] = note

print("KNN-based imputation complete.")

# Create numeric + embedding combined feature space
X = np.hstack([
    embeddings,
    claims_df[["claim_amount"]].fillna(claims_df["claim_amount"].median()).values.reshape(-1, 1)
])

# Label encoding
claims_df["status"] = claims_df["status"].astype(str)
y = claims_df["status"]

# Only train on rows where status exists
df_train = claims_df[claims_df["status"] != "nan"]
X_train = X[claims_df["status"] != "nan"]
y_train = y[claims_df["status"] != "nan"]

X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

clf = RandomForestClassifier(n_estimators=200, class_weight="balanced")
clf.fit(X_train_split, y_train_split)

y_pred = clf.predict(X_test_split)

print(classification_report(y_test_split, y_pred))

missing_status_mask = (claims_df["status"] == "nan")

if missing_status_mask.sum() > 0:
    claims_df.loc[missing_status_mask, "status"] = clf.predict(X[missing_status_mask]) #94% accuracy

# ---------- STANDARDIZE CATEGORICAL VALUES ---------- #

# Force lowercase for normalization
def categorical_to_standard():
    claims_df["status"] = claims_df["status"].str.strip().str.lower()
    claims_df["diagnosis"] = claims_df["diagnosis"].str.strip().str.lower()
    claims_df["denial_reason"] = claims_df["denial_reason"].astype(str).str.strip().str.lower()
    patients_df["gender"] = patients_df["gender"].astype(str).str.strip().str.lower()
    providers_df["specialty"] = providers_df["specialty"].astype(str).str.strip().str.lower()
    
    # Status normalization mapping
    status_map = {
        "approved": "Approved",
        "approve": "Approved",
        "completed": "Approved",
        "denied": "Denied",
        "declined": "Denied",
        "reject": "Denied",
        "pending": "Pending",
        "in review": "Pending",
        "processing": "Pending",
        "appealed": "Appealed",
        "reopened": "Appealed",
    }
    
    claims_df["status"] = claims_df["status"].replace(status_map)
    
    # Diagnosis normalization mapping (expandable based on ICD10)
    diagnosis_map = {
        "dm2": "Type 2 Diabetes",
        "t2d": "Type 2 Diabetes",
        "diabetes mellitus": "Type 2 Diabetes",
        "diabetes": "Type 2 Diabetes",
        "hypertension": "Hypertension",
        "high bp": "Hypertension",
        "htn": "Hypertension",
        "asthma": "Asthma",
        "covid": "COVID-19",
        "covid-19": "COVID-19",
        "sars cov 2": "COVID-19",
        "heart disease": "Heart Disease",
        "cad": "Heart Disease",
        "cardiac condition": "Heart Disease",
        "cancer": "Cancer",
        "oncology case": "Cancer"
    }
    
    claims_df["diagnosis"] = claims_df["diagnosis"].replace(diagnosis_map)
    
    # Normalize denial reasons (optional controlled labels)
    denial_reason_map = {
        "missing docs": "Missing documentation",
        "missing documentation": "Missing documentation",
        "incorrect coding": "Incorrect coding",
        "coding error": "Incorrect coding",
        "not covered": "Not covered service",
        "not covered service": "Not covered service",
        "duplicate": "Duplicate claim",
        "duplicate claim": "Duplicate claim",
        "eligibility issue": "Eligibility issue",
        "policy expired": "Policy expired"
    }
    
    claims_df["denial_reason"] = claims_df["denial_reason"].replace(denial_reason_map)
    
    # Gender normalization
    gender_map = {"m":"Male", "male":"Male", "f":"Female", "female":"Female", "other":"Other"}
    patients_df["gender"] = patients_df["gender"].replace(gender_map)
    
    # Specialty standardization
    specialty_map = {
        "cardiology": "Cardiology",
        "endocrinology": "Endocrinology",
        "general medicine": "General Medicine",
        "family physician": "Family Medicine",
        "pulmonology": "Pulmonology",
        "oncology": "Oncology"
    }
    
    providers_df["specialty"] = providers_df["specialty"].replace(specialty_map)
    
    print("Standardized categorical fields.")

categorical_to_standard()

# ---------- FORMAT NORMALIZATION ---------- #

# Clean currency-like values into float
def normalize_currency(value):
    if pd.isna(value):
        return np.nan
    # Remove currency symbols, commas, whitespace
    value = str(value).replace("₹", "").replace("$", "").replace(",", "").strip()
    try:
        return float(value)
    except:
        return np.nan

claims_df["claim_amount"] = claims_df["claim_amount"].apply(normalize_currency)


# ---- Normalize Dates to ISO Format (YYYY-MM-DD) ---- #
def normalize_date(value):
    if pd.isna(value):
        return np.nan
    try:
        # Auto-detect format using parser
        parsed = parser.parse(str(value), dayfirst=False)
        return parsed.strftime("%Y-%m-%d")   # ISO standard
    except:
        return np.nan

date_columns = ["service_date", "submission_date", "decision_date"]

for col in date_columns:
    claims_df[col] = claims_df[col].apply(normalize_date)


# --- Additional formatting: Trim whitespace in text fields --- #
text_columns = ["diagnosis", "status", "denial_reason", "notes"]

for col in text_columns:
    if col in claims_df.columns:
        claims_df[col] = claims_df[col].astype(str).str.strip()
        
claims_df = claims_df.drop("text_embedding_input", axis= 1)
print("Format normalization complete: currency + dates standardized.")
claims_df.head()
claims_df.to_csv("claims.csv", index=False)
# -----------------------
# SAVE RESULT
# -----------------------
# from pyspark.sql import SparkSession


# spark = SparkSession.builder \
#     .appName("ClaimsDBQuery") \
#     .getOrCreate()

# url = "jdbc:mysql://localhost:3306/claims_db"
# properties = {
#     "user": "sparkuser",
#     "password": "",
#     "driver": "com.mysql.cj.jdbc.Driver"
# }

# df = spark.read.jdbc(url=url, properties=properties)
# df.show()

# df.createOrReplaceTempView("claims_cleaned")




# from pyspark.sql import SparkSession
# try:
#     spark.stop()
# except:
#     pass

# mysql_connector_version = "9.5.0"
# mysql_coordinates = f"com.mysql:mysql-connector-j:{mysql_connector_version}"

# spark = SparkSession.builder \
#     .appName("ClaimsDBQuery") \
#     .config("spark.jars.packages", mysql_coordinates) \
#     .config("spark.jars.repositories", "https://repo1.maven.org/maven2/") \
#     .getOrCreate()

# spark.sparkContext._jvm.java.lang.Class.forName("com.mysql.cj.jdbc.Driver")

# url = "jdbc:mysql://localhost:3306/claims_db"
# properties = {
#     "user": "sparkuser",
#     "password": "",
#     "driver": "com.mysql.cj.jdbc.Driver"
# }

# df = spark.read.jdbc(url=url, table="claims_cleaned", properties=properties)
# df.show()



# ---------- Connect to SQLite ----------
db_path = "claims.db"
conn = sqlite3.connect(db_path)

# ---------- Create Table (if not exists) ----------
conn.execute("""
CREATE TABLE IF NOT EXISTS claims_cleaned (
    claim_id TEXT PRIMARY KEY,
    patient_id TEXT,
    provider_id TEXT,
    diagnosis TEXT,
    icd10_code TEXT,
    claim_amount REAL,
    service_date TEXT,
    submission_date TEXT,
    decision_date TEXT,
    status TEXT,
    denial_reason TEXT,
    notes TEXT
)
""")

# ---------- Save DataFrame into SQLite ----------
claims_df.to_sql("claims_cleaned", conn, if_exists="replace", index=False)

print("Data inserted successfully.")

# ---------- Read Back from Database ----------
result_df = pd.read_sql_query("SELECT * FROM claims_cleaned", conn)

print("\n Data Retrieved from Database:\n")
print(result_df)

conn.close()

