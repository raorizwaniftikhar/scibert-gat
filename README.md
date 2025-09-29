# 🧠 Hierarchical Multi-Label Classification with Graph and Contextual Embeddings

This repository implements a multi-stage hierarchical classification framework that combines semantic and structural information through SciBERT and Graph Attention Networks (GAT). The architecture is designed to improve classification accuracy across multi-level category hierarchies.

---

## 📌 Overview

The model consists of **five main stages**:

1. **Data Preparation**
2. **Contextual and Graph Embedding Creation**
3. **Feature Fusion via Gated Mechanisms**
4. **Dual Encoding and Classification**
5. **Evaluation with Hierarchical Metrics**

![Architecture Diagram](Proposed-Methodology.png)

---

## 1. 📂 Data Preparation

We preprocess the input text (titles and abstracts) before passing them through SciBERT.

### 🔧 Steps:

* **Text Cleaning**: Lowercasing, removing special characters, digits-only tokens, and excessive whitespaces.
* **SciBERT Tokenization**: Tokenization using WordPiece via the `BERTTokenizer` from SciBERT.

This ensures meaningful, model-friendly token inputs for downstream embedding.

---

## 2. 🔗 Graph Construction

Instead of word-level graphs, we build a **category-level graph** representing label hierarchies.

### 🌐 Category Graph:

* **Nodes**: Labels from three hierarchical levels (e.g., `cat1 → cat2 → cat3`)
* **Edges**: Directed edges encode parent-child relationships.

### 🧠 Graph Embeddings:

A **Graph Attention Network (GAT)** is applied to learn structural label embeddings that reflect inter-label dependencies.

---

## 3. 🧬 Graph Attention Network (GAT)

GAT models hierarchical label relationships through attention-based message passing.

### 📋 Key Steps:

* **Node Initialization**: Each label is initialized with a trainable vector.
* **Attention Mechanism**: Determines the relevance of neighboring nodes.
* **Message Passing**: Aggregates information from neighbors based on learned attention weights.
* **Stacking Layers**: Deepens understanding of graph structure across layers.

These embeddings capture **semantic and structural dependencies**, enriching classification.

---

## 4. ✨ SciBERT Contextual Embeddings

SciBERT is used to generate **contextual embeddings** from preprocessed titles and abstracts.

### 📌 Workflow:

* Input: Preprocessed text
* Output: Context Vectors (CVs) capturing deep semantic meaning

---

## 5. 🔁 Feature Fusion

We integrate contextual and structural embeddings through a **gated fusion mechanism**.

### 🧪 Fusion Components:

* **CV**: Context Vector from SciBERT
* **GV**: Graph Vector from GAT
* **Gated Fusion**: A gating layer learns to scale and combine both vectors optimally.

This enhances the representational power of input features for downstream classification.

---

## 6. 🏗️ Encoding Layer

The fused features undergo **dual encoding**:

* **Transformer Encoder**: Captures intra-document relationships (sentence-level).
* **Hierarchical Attention Network (HAN)**: Extracts document-level semantics by attending to important sentence features.

---

## 7. 🧮 Classification

* Final output from HAN is passed through a **fully connected layer** with **sigmoid activation** for multi-label classification across all 3 levels.
* Ensures **hierarchical coherence**—child categories are only predicted if their parent category is active.

---

## 8. 📏 Evaluation Metrics

We use both **standard and hierarchical** evaluation metrics.

### 📐 Hierarchical Metrics:

Let `Tᵢ` be the set of true labels (including ancestors) and `Pᵢ` the predicted labels for the *i-th* sample.

* **Hierarchical Precision**
  [
  HP = \frac{1}{N} \sum_{i=1}^{N} \frac{|Pᵢ \cap Tᵢ|}{|Pᵢ|}
  ]

* **Hierarchical Recall**
  [
  HR = \frac{1}{N} \sum_{i=1}^{N} \frac{|Pᵢ \cap Tᵢ|}{|Tᵢ|}
  ]

* **Hierarchical F1 Score**
  [
  HF1 = \frac{2 \cdot HP \cdot HR}{HP + HR}
  ]

### ✅ Other Metrics:

* **Subset Accuracy**:
  [
  SA = \frac{1}{N} \sum_{i=1}^{N} 1(Pᵢ = Tᵢ)
  ]

* **Hamming Loss**:
  [
  HL = \frac{1}{N \cdot L} \sum_{i=1}^{N} \sum_{j=1}^{L} 1(y_{i,j} \ne \hat{y}_{i,j})
  ]

---

## 9. 🧪 Testing

During inference, the same pipeline is followed:

* Preprocessing
* Embedding via SciBERT
* GAT-based structural learning
* Feature fusion
* Encoding and classification

All trained parameters are used without gradient updates.

---

## 📁 Folder Structure

```
├── data/                # Raw and processed data
├── models/              # Saved model weights
├── src/                 # Source code
│   ├── final-model.ipynb
├── images/
│   └── Proposed-Methodology.png
├── README.md
```
