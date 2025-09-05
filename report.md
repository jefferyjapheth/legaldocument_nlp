Here’s a crisp, side-by-side readout plus what it means and what I’d change next.

# Head-to-head results (your numbers)

| Model                                       |  Accuracy |  Macro F1 | Weighted F1 |
| ------------------------------------------- | --------: | --------: | ----------: |
| **LegalBERT → sentence avg → LR**           | **0.692** | **0.381** |   **0.608** |
| **n-grams / CountVectorizer / TF-IDF → LR** | **0.870** | **0.833** |   **0.875** |

Big deltas: +17.8 pts accuracy, +45.2 pts macro-F1, +26.7 pts weighted-F1 in favor of TF-IDF+LR.

Per-class highlights:

* **LegalBERT+LR**: several classes have **precision/recall/F1 = 0.0** → the classifier **never predicted** those labels.
* **TF-IDF+LR**: strong across the board; one class (“7.0” in your report) is clearly the hardest (F1 ≈ 0.47), while others are high (0.82–0.95).

> Note: the TF-IDF report uses numeric labels (0.0–7.0). If “7.0” maps to the **catch-all/NA** class in your schema, this pattern (lowest F1) is common for heterogeneous “other/NA” buckets.

---

# Why did LegalBERT underperform so much?

Zero scores across multiple labels are a red flag for **pipeline/labeling issues**, not just model capacity. Most likely culprits:

1. **Label indexing consistency**
   If you fit a new `StringIndexer` on validation/test, label→index mappings can shift; any metric code that expects the training mapping will appear “off” and the classifier may effectively “miss” classes.
   ✅ Fit the indexer **once on train**, reuse that **same fitted model** everywhere:

   ```python
   label_indexer = StringIndexer(inputCol="type_label", outputCol="label_index")
   label_indexer_model = label_indexer.fit(train_ready)
   train_ready = label_indexer_model.transform(train_ready)
   val_ready   = label_indexer_model.transform(val_ready)
   test_ready  = label_indexer_model.transform(test_ready)
   ```

2. **Embedding pooling + finisher shape**

   * You’re doing **token BERT → average pooling → EmbeddingsFinisher → take `[0]`**.
   * If a doc has multiple sentences and you didn’t run a sentence detector, the finisher output shape can be surprising (e.g., array of sentence embeddings). Taking `[0]` may keep only the **first sentence**, discarding the rest.
   * **Fixes**:

     * Add a **SentenceDetector** (or SentenceDetectorDL) so you know exactly what you’re averaging.
     * Prefer **BertSentenceEmbeddings** with **`CLS` pooling** (often better than token-average for classification).
     * Set `EmbeddingsFinisher(...).setOutputAsVector(True)` and **drop the UDF** for cleaner, faster MLlib interop.

3. **Underfitting via regularization / features**
   Your LR uses `regParam=0.1` + `elasticNetParam=0.1`. That’s fairly strong shrinkage on dense, \~768-dim vectors and can collapse minority classes. Try `regParam` in `{0.001, 0.01, 0.03}` and consider `elasticNetParam=0.0` (pure L2) first.

4. **Weights column**
   You set `weightCol="class_weight"`. If that column is missing/constant or incorrectly computed, you can skew learning. Verify it exists and the weights sum sensibly across classes; otherwise drop `weightCol` for a baseline.

5. **Model choice**
   LR on frozen embeddings is fine, but **end-to-end** methods (e.g., Spark NLP **ClassifierDLApproach** over sentence embeddings, or a **LegalBERT sequence classifier**) usually win on semantics.

---

# Complexity & scalability trade-offs

**n-grams / TF-IDF + LR**

* **Performance**: Excellent here (Accuracy 0.87, Macro-F1 0.83). In domains where labels correlate with keywords (“lease”, “employment”, “shareholder”), sparse BoW shines.
* **Complexity**: Simple to build, easy to debug. Hyperparams (ngram range, minDF, regParam) are straightforward.
* **Scalability**: Linear, memory-bound by vocab size. Spark scales this very well; consider `HashingTF` to cap memory. Inference is **very fast**.

**LegalBERT embeddings + LR (frozen)**

* **Performance**: Should help when phrasing shifts or context matters; in your run it underperformed (likely pipeline issues + pooling). With the right setup it can match/beat TF-IDF on semantically tricky cases.
* **Complexity**: Higher. You manage tokenization, sentence segmentation, pooling, and vector plumbing.
* **Scalability**: **Slower** and costlier (GPU helps for embedding). Inference latency and throughput are worse than TF-IDF. However, feature dimensionality is bounded (e.g., 768), which can be nice operationally.

**End-to-end transformer (fine-tuned LegalBERT / ClassifierDL)**

* **Performance**: Often best when you have enough data (you do) and diverse phrasing.
* **Complexity**: Higher training cost; simpler serving (single model).
* **Scalability**: Training benefits from GPU; serving is heavier than TF-IDF but can be batched.

---

# Which categories are harder & why

From your TF-IDF report:

* **Class “7.0”** has the **lowest F1 (\~0.47)**. If that’s your **NA/Other** bucket, it’s expected: it’s **heterogeneous**, lacks consistent keywords, and overlaps lexically with everything else.
* The next weakest appears around **0.77–0.82 F1** (e.g., the class with F1≈0.819). In contract taxonomies, this is typically **services & supply** or similar—broad, overlapping vocabulary (“service”, “agreement”, “supplier”, “term”) that also appears in **employment** or **purchase & M\&A** templates.

Why these struggle:

* **Vocabulary overlap** between neighboring classes.
* **Boilerplate sections** shared across agreement types (warranty, indemnity, governing law).
* **Short/partial documents** or **headers only** in some samples.
* **Label noise**: “NA/Other” mixtures or mismapped documents.

---

# What I’d change (actionable)

**1) Fix/strengthen the LegalBERT pipeline**

```python
# 1) Add sentence detection
sentencer = SentenceDetectorDLModel.pretrained() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

# 2) Use sentence-level BERT with CLS pooling
bert_sent = BertSentenceEmbeddings.load("models/legalbert_sentence_emb") \
    .setInputCols(["sentence"]) \
    .setOutputCol("sentence_embeddings") \
    .setCaseSensitive(False) \
    .setPoolingLayer("CLS")   # instead of token average

# 3) Finish as ML vector (no UDF)
emb_fin = EmbeddingsFinisher() \
    .setInputCols(["sentence_embeddings"]) \
    .setOutputCols(["features"]) \
    .setOutputAsVector(True) \
    .setCleanAnnotations(True)

nlp_pipeline = Pipeline(stages=[document, sentencer, bert_sent, emb_fin])
```

Then tune LR with smaller `regParam` and **no** `weightCol` to start; re-introduce weights only if needed.

**2) Try end-to-end classification (stronger baseline)**

* Spark NLP **ClassifierDLApproach** on sentence embeddings, or a **LegalBERTForSequenceClassification** model. This lets the model learn class boundaries directly rather than relying on a linear head over frozen features.

**3) Keep TF-IDF+LR as a production baseline**

* It’s fast and currently best. Hardening tips:

  * Add **char n-grams** (captures legal abbreviations, section codes).
  * Tune `minDF`, `ngramRange=(1,2 or 3)`, and `regParam`.
  * Consider **HashingTF** for memory predictability at scale.

**4) Data + label hygiene**

* Audit the **NA/Other** class. Split into clearer subtypes or shrink its scope.
* Ensure **stratified splits** and consistent `StringIndexerModel`.
* For the hardest class, curate a small batch of **hard-negative** examples and oversample or weight during training.

**5) Evaluation you already have (good)**

* Report **accuracy + macro-F1 + weighted-F1** (done).
* Keep the **confusion matrix** (you’re logging it). Use it to quantify the top confusions (e.g., “class A → class B” counts) and target data fixes.

---

## TL;DR

* Today, **TF-IDF+LR wins decisively** on your data (+17.8 acc, +45.2 macro-F1).
* The **LegalBERT pipeline likely has setup issues** (label indexing / pooling / vector plumbing) causing several classes to never be predicted.
* Fix the pipeline (sentence detection + CLS pooling + consistent indexer + lighter regularization) or go **end-to-end** with ClassifierDL / LegalBERT fine-tuning.
* Expect the **NA/Other-like** class to remain the hardest; address it with label hygiene and targeted data curation.
