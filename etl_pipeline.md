# Contract Classification Data Pipeline – Documentation

## Problem Statement

Our organization manages thousands of legal and business contracts (e.g., NDAs, SLAs, Employment Contracts, Vendor Agreements, Partnership Agreements).
Manually sorting and tagging these contracts is **slow, error-prone, and costly**.
We aim to build an **AI-powered classifier** to automatically categorize contracts into their correct types.

---

## Dataset Preparation

### Source Data

* Contracts are stored as raw text (`description`) with metadata (`contract` path).
* Each contract has an **initial machine-assigned label** (`type_label`) and a **confidence score** (`type_score`).

### Preprocessing Steps

1. **Relabeling `agreement_type`**

   * Mapped misnamed `type_label` values into human-readable `agreement_type`:

     * `LABEL_0 → security`
     * `LABEL_1 → employment`
     * `LABEL_2 → lease`
     * `LABEL_3 → services&supply`
     * `LABEL_4 → purchase&ma`
     * `LABEL_5 → shareholder`
     * `LABEL_6 → other`
     * `LABEL_7 → na`

2. **Extracted Contract Name**

   * From the `contract` path, extracted the file name using regex.
   * Reassigned it as `contract` for readability.

3. **Filtering**

   * Removed contracts with very short descriptions (`LENGTH(TRIM(description)) <= 20`).
   * Prepared **two datasets**:

     * **Full dataset** (8 categories: includes `na` and `other`).
     * **Filtered dataset** (6 core categories: excludes `na` and `other`).

---

##  Dataset Statistics

### Full Dataset

| Agreement Type    | Count   | Percentage |
| ----------------- | ------- | ---------- |
| Employment        | 144,489 | 39.35%     |
| Security          | 102,328 | 27.87%     |
| Purchase & M\&A   | 47,850  | 13.03%     |
| Services & Supply | 24,882  | 6.78%      |
| Shareholder       | 15,716  | 4.28%      |
| Other             | 13,218  | 3.60%      |
| NA                | 10,142  | 2.76%      |
| Lease             | 8,584   | 2.34%      |

* Total **211,636 distinct contracts**.

### Filtered Dataset (Production-Ready)

* Contains only **6 meaningful categories**:
  `employment, security, purchase&ma, services&supply, shareholder, lease`.
* `na` and `other` dropped for cleaner training.
* At inference, **low-confidence predictions** (low `type_score`) can be mapped back to `"other"`.

---

##  Data Pipeline Output

### Views Created

* `relabeled_contracts` → Relabeled data with agreement types.
* `relabeled_contracts_named` → Added `contract_name`.
* `relabeled_contracts_final` → Cleaned dataset ready for saving.
* `relabeled_contracts_filtered` → Production-ready training dataset.

### Saved Outputs

* `data/processed/mcc_contracts_full` → Parquet with **all 8 categories**.
* `data/processed/mcc_contracts` → Parquet with **6 categories** (for training).

Both are partitioned by `type_label`.

---

##  Next Steps

1. **Model Training**

   * Use `description` as main input text.
   * Try both **classical ML** (TF-IDF + Logistic Regression/SVM) and **transformers** (BERT, LegalBERT).
   * Compare performance with Accuracy, F1-score, and Confusion Matrix.

2. **Handling Edge Cases**

   * Map low `type_score` predictions to `"other"`.
   * Consider multi-stage classification (high-level filter + fine-grained classification).

3. **Deployment**

   * Build inference pipeline using the **filtered model**.
   * Provide human-readable `agreement_type` + `contract_name` in outputs.

---

 **Summary**:
We now have a **clean, production-ready dataset** with 6 core contract types, saved in parquet format and partitioned for scalability. This dataset is ready to be used for training both baseline ML models and advanced transformer-based classifiers.



**Detailed commit / PR description (for merges):**
feat(data-pipeline): add relabeled and filtered contract datasets

- Relabeled contract type_label → agreement_type for readability
- Extracted contract_name from file path
- Created relabeled_contracts_final (full dataset, 8 categories)
- Created relabeled_contracts_filtered (production-ready dataset, 6 categories)
- Saved both datasets as partitioned Parquet files under data/processed
- Prepared clean training data for NLP classification
