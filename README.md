# FriendsRAG 📺

A Retrieval-Augmented Generation (RAG) system that helps Friends fans find episodes based on scene descriptions. Simply describe any scene from the show, and FriendsRAG will identify the episode and provide a complete summary!

**Example:**
- **You ask:** "What's the episode where Rachel gets off the plane?"
- **FriendsRAG returns:** Episode title, season, episode number, and full summary

## 📁 Repository Structure

```
FriendsRAG/
├── RAG.ipynb                # Core RAG implementation
├── RAG_Langfuse.ipynb       # RAG with Langfuse monitoring
├── RAG_FastAPI.py           # FastAPI endpoint
├── requirements.txt         # Python dependencies
└── data/
    ├── FriendsEpisodes.csv  # Episode metadata
    └── FriendsScripts.csv   # Episode scripts
```

## 🚀 Quick Start

### 1. Install Dependencies

**Using uv (recommended):**
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

**Using pip:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure AWS Credentials

Login with AWS SSO:
```bash
aws sso login --profile your-profile-name
```

Update the AWS configuration in each file:
```python
REGION_NAME = "us-east-1"  # Your AWS region
CREDENTIALS_PROFILE_NAME = "your-profile-name"  # Your AWS profile
```

## 📖 Usage Guide

### RAG.ipynb - Core Implementation

**What it does:** Creates the vector database and allows you to query episodes interactively.

**Requirements:**
- AWS credentials configured
- Data files in `data/` folder

**How to run:**
```bash
jupyter notebook RAG.ipynb
```

**Run all cells to:**
1. Load episode and script data
2. Create embeddings and build FAISS vector database
3. Query the system with natural language

---

### RAG_Langfuse.ipynb - With Monitoring

**What it does:** Same as RAG.ipynb but adds Langfuse monitoring and evaluation metrics.

**Requirements:**
- AWS credentials configured
- Data files in `data/` folder
- Langfuse account and API keys

**Setup Langfuse:**
```bash
# Create a .env file with:
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_BASE_URL = "https://cloud.langfuse.com"
```

**How to run:**
```bash
jupyter notebook RAG_Langfuse.ipynb
```

**Features:**
- Request/response tracing
- Evaluation metrics (correctness, relevance, groundedness)
- Dashboard at https://cloud.langfuse.com

---

### RAG_FastAPI.py - API Endpoint

**What it does:** Provides a REST API to query the RAG system.

**Requirements:**
- AWS credentials configured
- **Vector database already created** (run RAG.ipynb first to generate `./vector_database/`)

**How to run:**
```bash
fastapi dev RAG_FastAPI.py
```

**API will be available at:** `http://localhost:8000`

**Test the API:**

Interactive docs:
```
Visit: http://localhost:8000/docs
```
