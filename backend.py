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

#Indexing + RAG Preparation 

claims_df = pd.read_csv("C:\\Users\\ASUS\\Downloads\\Abacus Insights - RAG + LLM\\Abacus Insights - RAG + LLM\\claims.csv")
#-------CHUNKING------#
def serialize_claim(row):
    return (
        f"Claim ID: {row['claim_id']}. "
        f"Patient: {row['patient_id']} treated by provider {row['provider_id']}. "
        f"Diagnosis: {row['diagnosis']}. "
        f"Status: {row['status']} with denial reason: {row['denial_reason']}. "
        f"Amount: {row['claim_amount']} billed. "
        f"Service date: {row['service_date']}."
    )
db_path = "claims.db"
conn = sqlite3.connect(db_path)
db_claims_df = pd.read_sql_query("SELECT * FROM claims_cleaned", conn)
db_claims_df["search_text"] = db_claims_df.apply(serialize_claim, axis=1)
claims_df["search_text"] = claims_df.apply(serialize_claim, axis= 1)
db_claims_df.to_sql("claims_cleaned", conn, if_exists="replace", index=False)
claims_df.to_csv("claims.csv", index=False)
conn.close()


#Query and Retrieval 

print(claims_df.head())


# -------------------------------
# 1. Configuration
# -------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


ALLOWED_SQL_KEYWORDS = {
    "SELECT", "FROM", "WHERE", "AND", "OR", "LIKE", "BETWEEN", "LIMIT",
    "ORDER", "BY", "DESC", "ASC", "=", "<", ">", "<=", ">=", "!=",
    "IN", "NOT", "IS", "NULL", "EXTRACT"
}

ALLOWED_TABLES = {"claims_cleaned"}

ALLOWED_COLUMNS = {
    "claim_id", "status", "claim_amount", "diagnosis", "claim_date", "insurer"
}

# -------------------------------
# 2. LLM → Convert Query to SQL
# -------------------------------

def generate_sql_from_query(user_query: str) -> str:
    client = Groq(api_key=GROQ_API_KEY)

    prompt = f"""
        You are a strict SQL generator.
    
        Database Schema:
        Table: claims_cleaned
        Columns:
        - claim_id (TEXT)
        - patient_id (TEXT)
        - provider_id (TEXT)
        - diagnosis (TEXT)
        - icd10_code (TEXT)
        - claim_amount (FLOAT)
        - service_date (DATE)
        - submission_date (DATE)
        - decision_date (DATE)
        - status (TEXT)
        - denial_reason (TEXT)
        - notes (TEXT)
        
        RULES:
        - Only generate SQL for the schema provided
        - No DROP, ALTER, DELETE, INSERT, or UPDATE statements allowed
        - Do NOT define an alias for a column or a table.
        - Do NOT hallucinate non-existent tables or fields
        - Only return the SQL query. No explanations.
        
        Examples:
        Q: "show approved claims"
        A: SELECT * FROM claims_cleaned WHERE status = 'Approved';
        
        Q: "find claims less than 2000"
        A: SELECT * FROM claims_cleaned WHERE claim_amount < 2000;
        
        Now convert:
        "{user_query}"

    """

    response = client.chat.completions.create(
        model= "openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}]
    )

    sql_query = response.choices[0].message.content.strip()

    return sql_query


# -------------------------------
# 3. SQL Safety Validation
# -------------------------------

def validate_sql(sql: str):
    parsed = sqlparse.parse(sql)[0]

    # Check forbidden keywords like DELETE, DROP, etc.
    forbidden = {"DELETE", "DROP", "INSERT", "UPDATE", "TRUNCATE", "ALTER"}
    if any(token.value.upper() in forbidden for token in parsed.tokens):
        raise ValueError("Unsafe SQL detected — forbidden command present.")

    # Validate allowed SQL keywords
    for token in parsed.tokens:
        if token.is_keyword and token.value.upper() not in ALLOWED_SQL_KEYWORDS:
            raise ValueError(f"Invalid or dangerous SQL keyword: {token.value}")

    # Validate table usage
    if not any(tbl in sql for tbl in ALLOWED_TABLES):
        raise ValueError("SQL references unknown or unauthorized table(s).")

    # Validate columns
    for word in sql.replace(",", " ").replace(";", "").split():
        if "." not in word and word in ALLOWED_COLUMNS:
            continue

    return True


# -------------------------------
# 4. Execute the Query Safely
# -------------------------------

def execute_sql(sql: str):

    db_path = "claims.db"
    conn = sqlite3.connect(db_path)
    try:
        # Execute query using Spark SQL engine
        result_df = pd.read_sql_query(sql, conn)
        conn.close()
        return result_df

    except Exception as e:
        return pd.DataFrame({"error": str(e)})

# -------------------------------
# 5. Full Pipeline Function
# -------------------------------

def process_user_query(user_query: str):
    print("\n User Query:", user_query)

    sql = generate_sql_from_query(user_query)
    print("\n Generated SQL:", sql)

    validate_sql(sql)

    print("\n SQL Validation: PASSED")

    results = execute_sql(sql)

    print("\n Results:", results)
    return results

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


def rerank_with_faiss(query, df, k=10):
    
    text_data = df["search_text"].tolist()
    query_emb = embed_text([normalize_query(query)]).cpu().numpy()
    subset_embs = embed_text(text_data).cpu().numpy()
    
    index = faiss.IndexFlatIP(query_emb.shape[1])
    index.add(subset_embs)

    distances, indices = index.search(query_emb, k=min(k, len(df)))
    
    return df.iloc[indices[0]].assign(similarity=distances[0])




def dataframe_to_context(df, limit=10):
    """
    Converts a DataFrame of insurance claims into a readable text block
    for the LLM to reconstruct meaning from.

    Args:
        df: Pandas DataFrame containing claims.
        limit: Max number of rows to include (prevents huge context).

    Returns:
        A formatted multi-line string for prompt insertion.
    """
    required_columns = [
    "claim_id", "diagnosis", "status", "claim_amount",
    "service_date", "decision_date", "denial_reason"]

    for _, row in df.head(limit).iterrows():
        # Build row text dynamically with fallbacks
        row_text_parts = []

        for col in required_columns:
            if col in df.columns:
                value = row.get(col, "N/A")
            else:
                value = "N/A"   # Column missing entirely

            row_text_parts.append(f"{col.replace('_', ' ').title()}: {value}")

        row_text = ", ".join(row_text_parts)
        print(row_text)
        return row_text


def generate_answer_groq(query, df):
    
    client = Groq(api_key=GROQ_API_KEY)
    if df.empty:
        return
    
    system_msg = """
    You are an insurance claims assistant AI. Your job is to summarize retrieved claims
    and explain findings clearly and professionally. Use bullet points when useful.
    """
    
    context = dataframe_to_context(df)

   
        

    user_prompt = f"""
    User Query: "{query}"

    Retrieved Claim Records:
    {context}

    Instructions:
    - Summarize results in natural language.
    - If counts, totals, or notable patterns exist, mention them.
    - Keep response under 200 words.
    - If the Retrieved Claim Records is empty, highlight that
    """

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",   # Best model for reasoning
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content

# controlled vocab mapping from earlier standardization step
status_map = {
    "denied": "Denied",
    "approved": "Approved",
    "pending": "Pending",
    "appealed": "Appealed"
}

def normalize_query(query: str):
    query = query.lower().strip()
    structured_parts = []

    # 1. Status Extraction
    for word, std in status_map.items():
        if word in query:
            structured_parts.append(f"Status: {std}.")
            break

    # 2. Numeric Comparison Extraction
    if match := re.search(r"less than (\d+)", query):
        structured_parts.append(f"Claim amount < {match.group(1)}.")

    if match := re.search(r"greater than (\d+)", query):
        structured_parts.append(f"Claim amount > {match.group(1)}.")

    if match := re.search(r"between (\d+) and (\d+)", query):
        structured_parts.append(f"Claim amount between {match.group(1)} and {match.group(2)}.")

    # 3. Diagnosis Extraction (semantic keywords)
    for condition in ["diabetes", "asthma", "hypertension", "covid", "cancer", "heart"]:
        if condition in query:
            structured_parts.append(f"Diagnosis: {condition.title()}.")
            break

    # If no parts detected, fallback to original query
    if not structured_parts:
        return f"Query: {query}"

    # Combine into normalized format
    return " ".join(structured_parts)

def run_user_query(user_query: str):
    normalized_query = normalize_query(user_query)
    results_df = process_user_query(user_query)
    final_output = "No relevant claims found."
    fin_df = pd.DataFrame()
    if not results_df.empty and "search_text" in results_df.columns:
        fin_df = rerank_with_faiss(normalized_query, results_df)
    if results_df.empty and "search_text" in claims_df.columns:
        fin_df = rerank_with_faiss(normalized_query, claims_df)
    print(results_df, "HERE")
    if not results_df.empty:
        final_output = generate_answer_groq(user_query, fin_df)
        print(final_output)

    return final_output