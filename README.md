# 📚 BiblioMatch — Intelligent Book Recommendation System

> An intelligent book recommendation system built on the Goodbooks-10k dataset that demonstrates end-to-end AI/ML concepts — from knowledge representation and EDA to deep learning, explainability (XAI), and responsible AI — across 10 structured lab tasks.

---

## 🚀 Overview

**BiblioMatch** is a Streamlit-based interactive web application that walks through the complete lifecycle of building an AI-powered book recommender. It uses the [Goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k) dataset and covers 10 structured lab tasks mapped to course outcomes (CO) and Bloom's Taxonomy cognitive levels.

---

## 🗂️ Task Overview

| Task | Title | CO | Cognitive Level |
|------|-------|----|----------------|
| Task 1 | Knowledge Representation & Ontology Mapping | CO1 | Remember / Understand |
| Task 2 | Reasoning & Inference Engine | CO1 | Understand / Apply |
| Task 3 | Search & Retrieval (BFS / DFS / A*) | CO2 | Apply / Analyze |
| Task 4 | Exploratory Data Analysis (EDA) | CO2 | Analyze |
| Task 5 | Feature Engineering & Preprocessing | CO2 | Apply / Analyze |
| Task 6 | Classical ML Models (KNN, SVD, Content-Based) | CO3 | Apply / Evaluate |
| Task 7 | Deep Learning Model (Neural Collaborative Filtering) | CO3 | Evaluate / Create |
| Task 8 | Explainability & XAI (LIME / SHAP) | CO4 | Evaluate / Analyze |
| Task 9 | End-to-End System Integration & Evaluation | CO4 | Evaluate / Create |
| Task 10 | Responsible AI Audit & Bias Mitigation | CO4 | Evaluate / Create |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly, Matplotlib, Seaborn |
| ML Models | Scikit-learn, Surprise (SVD) |
| Deep Learning | PyTorch (Neural CF) |
| Explainability | LIME, SHAP |
| Graph / Knowledge | NetworkX |
| Language | Python 3.11+ |

---

## 📁 Project Structure

```
Bibliomatch_mark2/
│
├── 10k_app.py              # Main Streamlit application entry point
├── task_components.py      # Modular task rendering components (Task 1–10)
├── logo.jpg                # App logo
│
├── 10k/                    # Goodbooks-10k dataset files
│   ├── books.csv
│   ├── ratings.csv
│   ├── book_tags.csv
│   ├── tags.csv
│   └── to_read.csv
│
├── 7k/                     # Alternate 7k subset dataset
│
├── Deep Learning Model 10k/ # Pre-trained NCF model artifacts
│
├── add_xai.py              # XAI (LIME/SHAP) integration script
├── fix_app.py              # App fix/patch script
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.11+
- pip

### Install Dependencies

```bash
pip install streamlit pandas numpy plotly matplotlib seaborn scikit-learn scikit-surprise torch torchvision networkx lime shap
```

### Run the Application

```bash
streamlit run 10k_app.py --server.port 8501
```

Then open your browser at: **http://localhost:8501**

---

## 🧭 Navigation

The sidebar lists **Task 1 through Task 10** in order. Click any task to jump to its interactive module, which includes:

- 📖 **Theory** — concept explanation
- 🔬 **Demo** — live interactive visualization / model run
- 📊 **Results** — metrics, charts, and insights
- 🎯 **Viva Q&A** — quick revision questions per task

---

## 📊 Dataset

**Goodbooks-10k** — 10,000 popular books with:
- ~6 million ratings by ~53,000 users
- Book metadata: title, author, genre tags, cover images
- Source: [goodbooks-10k on GitHub](https://github.com/zygmuntz/goodbooks-10k)

---

## 🎓 Course Outcome Mapping

| CO | Description |
|----|-------------|
| CO1 | Demonstrate knowledge representation and reasoning techniques in AI |
| CO2 | Apply search and data analysis methods to real-world problems |
| CO3 | Build and evaluate classical and deep learning recommendation models |
| CO4 | Integrate, explain, and audit AI systems for fairness and responsibility |

---

## 👤 Author

**Sanil Samir Mhatre**  
B.Tech — Artificial Intelligence & Data Science  

---

## 📄 License

This project is for academic purposes. Dataset credits to [Zygmunt Z.](https://github.com/zygmuntz/goodbooks-10k).
