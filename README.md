🧠 Claims Intelligence AI

A RAG-Powered Natural Language Query Assistant for Insurance Claims

📌 Overview

Claims Intelligence AI is an end-to-end intelligent assistant designed to help insurance staff retrieve, analyze, and understand claims using natural language queries instead of complex filters or SQL.
The system integrates:

ETL preprocessing

LLM-based reasoning

FAISS vector retrieval

Predictive ML models

A user-friendly chatbot interface

Users can ask questions like:

“Show me denied diabetes claims from last quarter.”
and receive meaningful, contextual responses grounded in real claims data.

🎯 Key Objectives

Enable insurance users to query claims using natural language.

Retrieve relevant claims data using semantic similarity (RAG).

Generate clear insights with LLMs based on real claim records.

Reduce manual workload and streamline claim analysis workflows.

🚀 Features

🔍 Natural Language Querying

📁 RAG-based Result Retrieval (FAISS + Embeddings)

🤖 LLM Response Generation

🧹 Realistic ETL with Missing Data Handling

🧠 RandomForestClassifier to Predict Missing Claim Status

💬 Web-based Chatbot UI

🗄️ SQLite/MongoDB Storage Support

🛠 Tech Stack
Stage	Tools
ETL & Processing	Python, Pandas, NumPy
Retrieval & Embeddings	Sentence Transformers, FAISS
ML Modeling	RandomForestClassifier (Scikit-Learn)
LLM Middleware	Groq
Backend	Flask / FastAPI
Frontend	HTML, CSS, JavaScript
Storage	SQLite 
🧬 Data Generation

Synthetic datasets emulate real-world insurance scenarios including:

Null values

Typos and noise

Inconsistent structures

Status variations (approved, denied, pending)

Handling includes:

Auto-cleanup with ETL

Routing corrupted rows to a review queue

Missing value imputation using FAISS nearest-neighbor vector averaging

🧩 System Architecture
📁 Data Source → 🧹 ETL → 🔤 Embeddings + FAISS Index 
    → 🤖 RAG + LLM → 💬 Chat Interface Output

▶️ Live Demo Flow

User enters a natural language question.

System retrieves relevant records using FAISS semantic search.

RAG passes context + user query to the LLM.

Assistant generates concise, structured insights.

Response displayed in interactive chat UI.

📈 Future Roadmap

Multi-organization deployment (SaaS model)

Real-time claim stream and audit alerts

Role-based access control

Analytics dashboards and visualization

Voice input and mobile app support

🏗 Setup & Run
# Clone repository
git clone <repo-url>
cd claims-ai

# Install dependencies
pip install -r requirements.txt

# Run backend
python app.py

# Open UI in browser:
http://localhost:5000


👤 Developer: Rehan Shahid Khna
📍 Project Type: RAG + ML + Full-Stack Prototype

📜 License

MIT License — free for academic and non-commercial use.
