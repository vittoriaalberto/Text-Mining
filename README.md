# ⚖️ Agentic RAG App - Setup Guide

### 0) Clone the repository in a folder and Go to project folder
Open your terminal and navigate to the project root:

```bash
cd path/to/agentic_rag_app
```

---

## 🛠️ PART 1: COMMON SETUP

Do these steps first, regardless of whether you use Docker or run locally.

### 1. Create `.env` file

Create a file named `.env` in the project root. Example content:

```ini
OPENAI_API_KEY=sk-your-real-key-here
HUGGINGFACEHUB_API_TOKEN=sk-your-real-key-here
```

### 2. Prepare Data Structure

Ensure your `Contest_Data` folder is organized exactly like this in your project root:

```text
agentic_rag_app/
├── Contest_Data/
│   ├── Italy/
│   │   ├── Divorce_Italy/
│   │   │   └── files.json
│   │   ├── Inheritance_Italy/
│   │   │   └── files.json
│   │   └── italian_cases_json_processed/
│   │       └── files.json
│   ├── Slovenia/
│   │   ├── Divorce_Slovenia/
│   │   │   └── files.json
│   │   ├── Inheritance_Slovenia/
│   │   │   └── files.json
│   │   └── slovenian_cases_json_processed/
│   │       └── files.json
│   └── Estonia/
│       ├── Divorce_Estonia/
│       │   └── files.json
│       ├── Inheritance_Estonia/
│       │   └── files.json
│       └── estonian_cases_json_processed/
│           └── files.json
```

---

## 🚀 PART 2: CHOOSE HOW TO RUN

### OPTION A: 🐍 Run Locally (No Docker)

**1. Create & activate virtualenv**

*Windows (PowerShell):*

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

*Linux / macOS:*

```bash
python -m venv .venv
source .venv/bin/activate
```

**2. Install dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**3. Run the App**

```bash
streamlit run app.py
```

---

### OPTION B: 🐳 Run with Docker

**1. Build Docker image**

```bash
docker build -t agentic-rag-app .
```

**2. Run Container**

*Linux / macOS:*

```bash
docker run -p 8501:8501 \
  --env-file .env \
  -v $(pwd)/Contest_Data:/app/Contest_Data \
  agentic-rag-app
```

*Windows (PowerShell):*

```powershell
docker run -p 8501:8501 `
  --env-file .env `
  -v ${PWD}/Contest_Data:/app/Contest_Data `
  agentic-rag-app
```


---

## 📱 PART 3: USAGE

Once the app is running, open **http://localhost:8501** in your browser.

**In the UI:**

* **Page 1 (Welcome page):** Explain how to use the Legal Chatbot app.
* **Page 2 (Chatbot Q&A):** Set configuration in the left sidebar and start asking questions.
* **Page 3 (Evaluation):** Evaluate queries.

---

ℹ️ Note
This repository is intended exclusively for educational purposes 🎓 and for personal study, within the scope of the TM course.
