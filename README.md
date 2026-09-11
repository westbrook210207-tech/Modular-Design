# Two-Tower Recommendation & FAISS Candidate Retrieval

A PyTorch implementation of a **Two-Tower (Dual-Encoder) Candidate Generation System** trained on the **MovieLens 100k** dataset and indexed using **FAISS** for fast Maximum Inner Product Search (MIPS).

---

## 📌 Project Overview

Candidate generation (retrieval) is the first stage of modern industrial recommendation systems. This project demonstrates how to learn joint vector representations for users and items, map them into a shared $32$-dimensional embedding space, and perform sub-millisecond retrieval across candidate pools using FAISS.

```
       ┌───────────────────┐               ┌───────────────────┐
       │   User Features   │               │   Item Features   │
       │ (Age, Gender, Occ)│               │  (Genres, Title)  │
       └─────────┬─────────┘               └─────────┬─────────┘
                 │                                   │
                 ▼                                   ▼
        ┌─────────────────┐                 ┌─────────────────┐
        │   User Tower    │                 │   Item Tower    │
        │   (2-Layer MLP) │                 │   (2-Layer MLP) │
        └─────────┬───────┘                 └─────────┬───────┘
                  │                                   │
                  ▼                                   ▼
          User Vector (32d)                  Item Vector (32d)
                  │                                   │
                  └───────────────┬───────────────────┘
                                  ▼
                     Dot Product / Cosine Sim
                                  │
                                  ▼
                           Interaction Prob
```

---

## 📁 Repository Structure

```text
movielens_twotower/
├── data/
│   └── ml-100k/            # Raw MovieLens 100k files (u.user, u.item, u.data)
├── src/
│   ├── __init__.py
│   ├── data.py             # Preprocessing, feature transformation & Leave-One-Out splitting
│   ├── models.py           # Dual-Encoder PyTorch architecture (User & Item Towers)
│   └── eval.py             # FAISS indexing & evaluation metrics (Recall, Precision, NDCG)
├── main.py                 # Pipeline execution script
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

---

## 🛠️ Key Features & Architecture

1. **Feature Engineering:**
   * **User Features:** Standardized continuous age, one-hot encoded gender and occupation.
   * **Item Features:** Multi-hot encoded 19 genre indicators.
2. **Dual-Encoder Architecture:**
   * Separate MLP towers projecting heterogeneous user and item feature spaces into a normalized $32$-dimensional embedding space.
3. **Training Objective:**
   * Binary Cross-Entropy Loss with negative sampling for implicit/explicit interaction prediction.
4. **FAISS Vector Indexing:**
   * Extracted item embeddings indexed into `faiss.IndexFlatIP` (Exact Inner Product Search).
5. **Evaluation Strategy:**
   * **Leave-One-Out Evaluation:** Evaluates candidates per user against held-out test interactions.
   * **Historical Filtering:** Filters out training set interactions before computing metrics across $K \in \{10, 20, 50, 100, 500, 1000\}$.
   * **Metrics:** Evaluates **Recall@K**, **Precision@K**, and **NDCG@K**.

---

## 🚀 Quick Start

### 1. Prerequisites & Setup

Clone the repository and install the dependencies:

```bash
git clone [https://github.com/your-username/movielens_twotower.git](https://github.com/your-username/movielens_twotower.git)
cd movielens_twotower
pip install -r requirements.txt
```

### 2. Dataset Preparation

Download and extract the [MovieLens 100k Dataset](https://grouplens.org/datasets/movielens/100k/):

```bash
mkdir -p data/ml-100k
# Move u.data, u.item, and u.user into data/ml-100k/
```

### 3. Run Pipeline

Train the model, build the FAISS index, and evaluate Top-$K$ candidate retrieval:

```bash
python main.py
```

---

## ⚙️ Requirements

* `torch >= 2.0.0`
* `faiss-cpu >= 1.7.4`
* `pandas >= 1.5.0`
* `numpy >= 1.22.0`
* `scikit-learn >= 1.0.0`

---

## 📊 Sample Metrics

| Metric | Top-10 | Top-50 | Top-100 | Top-500 |
| :--- | :---: | :---: | :---: | :---: |
| **Recall@K** | 0.0510 | 0.1805 | 0.2643 | 0.6582 
| **Precision@K** | 0.0051  | 0.0036 | 0.0020 | 0.0013 
| **NDCG@K** | 0.0234 | 0.0512 | 0.0648 | 0.1146 

~ 5X better than baseline
---

## 📜 License

Distributed under the MIT License.
