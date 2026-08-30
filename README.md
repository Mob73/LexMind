# ⚖️ Senunya — AI Legal Research Assistant for Togo

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B.svg)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange.svg)](https://www.trychroma.com/)
[![License](https://img.shields.io/badge/License-Educational-green.svg)](#-license)

An AI-powered legal research assistant designed to retrieve, analyze, and explain information from Togolese legal documents using **Retrieval-Augmented Generation (RAG)**.

---

## 📌 Overview

**Senunya** is an AI-powered legal research assistant focused on Togolese law.

The project combines a Retrieval-Augmented Generation (RAG) pipeline with a Large Language Model (LLM) to allow users to ask questions in natural language and retrieve relevant information from a collection of Togolese legal documents.

Instead of relying solely on the model's internal knowledge, Senunya first searches its legal document database, retrieves relevant passages, and then uses them as context to generate an accurate and grounded answer.

### 🎯 Goal
The objective is to make legal information easier to search, understand, and access, particularly for documents related to Togolese legislation.

---

##  Features

* 🔎 **Semantic legal search** — Retrieve passages based on context and meaning, not just exact keywords.
* 🤖 **AI-generated answers** — Natural language responses tailored to your specific query.
* 📚 **Retrieval-Augmented Generation (RAG)** — Grounded answers backed by indexed Togolese legal texts.
* ⚖️ **Togolese Law Focus** — Purpose-built context engine for regional legal documents.
* 🧩 **Document Chunking & Vectorization** — Optimized processing for long legal texts.
* 🗄️ **Persistent Storage** — Fast vector similarity search powered by ChromaDB.
* 🧠 **Multilingual Embeddings** — Flexible cross-lingual retrieval support.
* 💬 **Natural-Language Interaction** — Intuitive user experience for non-technical users.
* 🌐 **Streamlit Web Interface** — Lightweight, fast, and accessible web GUI.
* ⚠️ **Legal Disclaimer Integration** — In-app notices informing users of AI limits.

---

## 🧠 How Senunya Works

Senunya follows a structured RAG pipeline:

```text
                 User Question
                       │
                       ▼
              ┌─────────────────┐
              │ Query Processing│
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   Embeddings    │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │    ChromaDB     │
              │ Vector Search   │
              └────────┬────────┘
                       │
                Relevant Chunks
                       │
                       ▼
              ┌─────────────────┐
              │      LLM        │
              │  + Retrieved    │
              │    Context      │
              └────────┬────────┘
                       │
                       ▼
                Generated Answer

```

### 🔹 Step 1 — Document Processing

Legal documents are collected and processed before being added to the knowledge base. Large documents are divided into smaller chunks so that relevant sections can be retrieved efficiently.

### 🔹 Step 2 — Embeddings

Each chunk is transformed into a numerical vector using a multilingual embedding model. These vectors allow the system to perform semantic similarity searches rather than relying solely on exact keyword matching.

### 🔹 Step 3 — Vector Database

The generated embeddings are stored in ChromaDB. When a user asks a question, Senunya searches the vector database for the most relevant legal passages.

### 🔹 Step 4 — Retrieval

The most relevant chunks are selected and passed to the language model as contextual information.

### 🔹 Step 5 — Generation

The LLM uses the retrieved legal context to formulate a natural-language answer. This architecture helps significantly reduce hallucinations and ungrounded statements.

---

## 🏗️ Project Architecture

```text
Senunya/
│
├── src/                  # Core RAG components & utility modules
├── chroma_db/            # Persistent vector database store
├── app.py                # Streamlit web application entry point
├── main.py               # Main application logic / pipeline orchestration
├── chunks.pkl            # Processed document chunks cache
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation

```

---

## 🛠️ Tech Stack

| Technology | Purpose |
| --- | --- |
| **🐍 Python** | Core programming language |
| **🧠 LLM** | Natural-language answer generation (e.g., Google Gemini) |
| **🔎 RAG** | Retrieval + Generation architecture |
| **📚 ChromaDB** | Vector database for embedding storage & retrieval |
| **🔤 Multilingual Embeddings** | Semantic document and query vectorization |
| **🎨 Streamlit** | Interactive web application interface |
| **📄 Document Processing** | PDF parsing, chunking, and legal text ingestion |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone [https://github.com/Mob73/Senunya.git](https://github.com/Mob73/Senunya.git)
cd Senunya

```

### 2. Create and activate a virtual environment

**Windows:**

```cmd
python -m venv venv
venv\Scripts\activate

```

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate

```

### 3. Install dependencies

```bash
pip install -r requirements.txt

```

### 4. Configure environment variables

Create a `.env` file in the root directory and add your API credentials:

```env
GEMINI_API_KEY=your_api_key_here

```

> **Note:** Never commit your `.env` file or API keys to GitHub.

### 5. Run the application

```bash
streamlit run app.py

```

The application will launch automatically in your web browser at `http://localhost:8501`.

---

## 💬 Example Use Cases

Senunya can answer questions regarding Togolese law, such as:

* *"Quelles sont les conditions de formation d'un contrat ?"*
* *"Quels sont les droits du salarié en cas de licenciement ?"*
* *"Que prévoit le Code pénal togolais concernant cette infraction ?"*
* *"Quelles sont les règles applicables aux sociétés commerciales ?"*

---

## ⚠️ Limitations & Disclaimer

> **Important:** Senunya is an AI research tool and **should not** be considered a substitute for a qualified legal professional or official legal consultation.

* **Accuracy:** Answers generated by the system may contain errors, omissions, or context misinterpretations.
* **Knowledge Coverage:** The current legal database is evolving. Queries on unindexed texts may yield incomplete or unavailable responses.
* **Verification:** Always verify important legal information against official Togolese government sources and consult a licensed legal professional.

---

## 🔐 Security

API keys and sensitive configurations are managed via environment variables.

* Ensure `.env` is listed in your `.gitignore` file.
* Avoid hardcoding tokens or credentials into any tracked script.

---

## 🗺️ Roadmap

* [x] Legal document ingestion & preprocessing
* [x] Document chunking strategies
* [x] Multilingual embedding integration
* [x] Persistent vector database (ChromaDB)
* [x] Semantic retrieval pipeline
* [x] LLM context integration
* [x] Streamlit user interface
* [ ] Expand Togolese legal document repository (Codes, Decrees, Jurisprudence)
* [ ] Source citations & paragraph-level reference linkage
* [ ] Improved metadata filtering (by year, document type, jurisdiction)
* [ ] Advanced hybrid search (Keyword + Vector)
* [ ] RAG evaluation benchmarks (Faithfulness, Relevance metrics)
* [ ] Cloud deployment & production optimization

---

## 📊 Why RAG?

General-purpose LLMs often lack access to specific, regional, or recent legal documents. RAG addresses this by grounding the model with authoritative local documents.

```text
Standard LLM approach:
  User Question ──► General LLM ──► Generated Answer (Risk of hallucination)

Senunya RAG approach:
  User Question ──► Semantic Search ──► Relevant Togolese Legal Texts ──► LLM + Context ──► Grounded Answer

```

---

## 🎓 Project Purpose

Senunya is an experimental project exploring the intersection of:

* Artificial Intelligence & Natural Language Processing
* Retrieval-Augmented Generation (RAG) & Vector Databases
* Togolese Legal Information Accessibility

---

## 👨‍💻 Author

**Moubarak**

* GitHub: [@Mob73](https://www.google.com/search?q=https://github.com/Mob73)

---

## ⭐ Contributing

Contributions, issues, and feature requests are welcome!

Feel free to open an **Issue** or submit a **Pull Request** on the repository.

---

## 📄 License

This project is provided for educational, academic, and research purposes.

Please check the repository settings or contact the author for licensing terms.
