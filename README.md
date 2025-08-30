# Text Classification with Spark NLP

## Project Overview
Comparing classical ML and modern NLP approaches for legal document classification using Spark NLP.

## Approaches Implemented

### 1. Classical ML
- **TF-IDF + N-grams + CountVectorizer + Logistic Regression**
- Traditional feature engineering with statistical text representation

### 2. Hybrid Modern 
- **BERT + Logistic Regression**
- Pre-trained embeddings with classical ML classifier

### 3. Full Deep Learning
- **BERT + ClassifierDL**
- End-to-end neural classification with domain-specific LegalBERT embeddings

### 4. Alternative Deep Learning
- **USE + ClassifierDL** 
- Universal Sentence Encoder with neural classifier

## Evaluation Goals
- **Performance**: Accuracy, F1-score comparison across approaches
- **Complexity**: Model interpretability and hyperparameter requirements  
- **Scalability**: Training time, inference speed, and resource usage
- **Error Analysis**: Identify difficult document categories and classification patterns

## Expected Trade-offs
- **Classical**: Fast, interpretable, good baseline
- **Hybrid**: Better semantic understanding, moderate complexity
- **Deep Learning**: Highest accuracy potential, resource-intensive, less interpretable