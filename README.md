# Text Classification with Spark NLP

## Project Overview
Comparing classical ML and modern NLP approaches for legal document classification using Spark NLP.

## Approaches Implemented

### 1. Classical ML
- **TF-IDF + N-grams + CountVectorizer + Logistic Regression**
- Traditional feature engineering with statistical text representation

### 2. Full Deep Learning
- **LEGALBERT + ClassifierDL**
- End-to-end neural classification with domain-specific LegalBERT embeddings

### 3. Hybrid Modern 
- **LEGALBERT + Logistic Regression**
- Pre-trained embeddings with classical ML classifier


## Evaluation Goals
- **Performance**: Accuracy, F1-score comparison across approaches
- **Complexity**: Model interpretability and hyperparameter requirements  
- **Scalability**: Training time, inference speed, and resource usage
- **Error Analysis**: Identify difficult document categories and classification patterns

## Expected Trade-offs
- **Classical**: Fast, interpretable, good baseline
- **Hybrid**: Better semantic understanding, moderate complexity
- **Deep Learning**: Highest accuracy potential, resource-intensive, less interpretable


The purpose of your **N-grams + CountVectorizer + TF-IDF** approach is to create a **sophisticated classical ML baseline** that captures different levels of textual information through manual feature engineering:

## **Why This Approach:**

### **1. Comprehensive Feature Capture**
- **Unigrams**: Individual words ("contract", "legal", "payment")
- **Bigrams**: Word pairs ("legal document", "payment terms", "breach of")
- Captures both **individual concepts** and **contextual relationships**

### **2. Statistical Importance Weighting**
- **TF-IDF**: Highlights words that are frequent in a document but rare across the corpus
- Automatically identifies **discriminative terms** for each document category
- Reduces noise from common words

### **3. Domain-Specific Feature Engineering**
- Legal documents have specific **terminology patterns** and **phrase structures**
- Bigrams capture legal phrases like "force majeure", "intellectual property"
- Manual feature combination gives you **control** over what the model learns

### **4. Interpretable Baseline**
- You can **inspect the features** and understand what drives classifications
- **Vocabulary analysis** shows which terms/phrases are most important
- Provides **explainable results** for legal contexts where interpretability matters

### **5. Performance Benchmark**
- Establishes how well **traditional NLP techniques** work on your legal data
- Tests if **modern embeddings** (BERT) actually provide meaningful improvements
- **Cost-effective** approach that might be "good enough" without expensive deep learning

**Bottom line**: This approach tests whether careful traditional feature engineering can compete with modern pre-trained embeddings for legal document classification.
