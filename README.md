# **News Articles Classification: Detecting Fake News with AI**

### **Overview**
This project applies **deep learning** to classify news articles as **real or fake**, leveraging **CNN-Bidirectional LSTM** for advanced **feature extraction** and **sequence learning**. The model is trained on labeled news data to mitigate misinformation spread on digital platforms.

---

### **Project Workflow**
1. **Data Collection & Preprocessing**
   - Uses **'train.tsv'** for training and **'test.tsv'** for evaluation.
   - Target variable: **'label'** (1: Fake, 0: Real).
   - **Text preprocessing** includes stopword removal and token filtering using **NLTK & Gensim**.
   - **Tokenization & padding** ensure uniform input size.

2. **Model Architecture**
   - **Embedding Layer**: Converts words into dense vector representations.
   - **Convolutional Layers**: Extracts local text features using **Conv1D & MaxPooling**.
   - **Bidirectional LSTM**: Captures context from both past and future sequences.
   - **Dropout Layer**: Prevents overfitting.
   - **Dense Layer**: Outputs the probability of a news article being fake or real.

3. **Training & Evaluation**
   - **Loss Function**: Binary Cross-Entropy.
   - **Optimizer**: Adam.
   - **Metrics**: Accuracy, Precision, Recall, F1 Score, ROC AUC.
   - **Training Configuration**:
     - **5 epochs**, batch size **64**.
     - **80-20** train-validation split.
   - Achieved **97% accuracy** in detecting fake news.

4. **Future Work**
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
