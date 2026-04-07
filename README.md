# **News Articles Classification: Detecting Fake News with AI**

### **Overview**
This project leverages a **BERT-based transformer model** for news articles classification, utilizing contextual language understanding to accurately distinguish between **real and fake news articles**. The approach significantly improves semantic comprehension compared to traditional sequence models.

---

### **Project Workflow**
1. **Data Collection & Preprocessing**
   - Utilized 'train.tsv' (~100K+ samples) for training and 'test.tsv' for evaluation.
   - Target variable: 'label' (1: Fake, 0: Real).
   - Minimal preprocessing applied (removal of noisy tokens), preserving text structure for context-aware transformer learning.
   - Employed BERT tokenizer (WordPiece) for subword tokenization and dynamic padding.

3. **Model Architecture**
   - Implemented pretrained BERT (base-uncased) for deep contextual embeddings.
   - Added a classification head (dense layer + softmax) on top of BERT’s pooled output.
   - Fine-tuned the entire transformer model end-to-end for task-specific learning.
   - Leveraged self-attention mechanism to capture long-range dependencies and contextual relationships.

4. **Training & Evaluation**
   - **Loss Function**: Binary Cross-Entropy.
   - **Optimizer**: Adam.
   - **Metrics**: Accuracy, Precision, Recall, F1 Score, ROC AUC.
   - **Training Configuration**:
     - **5 epochs**, batch size **64**.
     - **80-20** train-validation split.
   - Achieved **97% accuracy** in detecting fake news.

5. **Future Work**
   - Extend to **multilingual datasets**.
   - Deploy model for **real-time news classification**.

---

### **Installation**
Ensure you have the required dependencies installed before running the code:

```bash
pip install pandas numpy tensorflow nltk gensim matplotlib
```

---

### **Usage**
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/news-classification.git
   cd news-classification
   ```

---

### **Results**
- **Model Performance Metrics**:
  - **Accuracy**: 97%
  - **Precision**: 97%
  - **Recall**: 97%
  - **F1 Score**: 97%
  - **ROC AUC Score**: 97%
- **Strong generalization** on unseen data.

---

### **Contributors**
- **Sachin Gaikwad**
- **Aarya Pawar**
- **Shikhar Kanauje**
- **Yash Dilip Phalke**

For any queries, feel free to open an issue.
