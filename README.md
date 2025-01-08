# 🚴 Cycling Project

Welcome to the **Cycling Project** repository! This project involves data cleaning, feature engineering, outlier detection, and model development for **classification** and **clustering** tasks related to cycling data.

---

## 🗂️ Repository Structure

### **Main Files**
1. **`Cycling_Project.pdf`**  
   The project report detailing the objectives, methodology, results, and conclusions.

2. **`1data_transformation.ipynb`**  
   - Handles general data cleaning and feature engineering.
   - Prepares the data for further analysis by defining core features.

3. **`2outlier_detection_pre.ipynb`**  
   - Focuses on identifying and handling outliers in the data.
   - Uses the features defined in the previous step.

4. **`3aggregation.ipynb`**  
   - Aggregates features for **classification** tasks.
   - Ensures no **data leakage** during feature engineering.

5. **`3aggregation_clustering.ipynb`**  
   - Aggregates features specifically for **clustering** tasks.

6. **`clustering.ipynb`**  
   - Implements the **clustering** analysis using the features prepared in `3aggregation_clustering.ipynb`.

7. **`pred/pred.ipynb`**  
   - Implements **classification**, testing, and evaluation.
   - Final predictions and results are generated here.

---

### **Input Data**
Place the following datasets in the `dataset/` folder at the root level:
- `stages.csv`
- `races.csv`

---

## 🔄 Workflow Order

Run the code in the following order for a seamless experience:

1. **Data Cleaning & Feature Engineering**  
   Run: `1data_transformation.ipynb`

2. **Outlier Detection**  
   Run: `2outlier_detection_pre.ipynb`

3. **Feature Aggregation**  
   - For **classification**: Run `3aggregation.ipynb`.  
   - For **clustering**: Run `3aggregation_clustering.ipynb`.

4. **Clustering Analysis**  
   Run: `clustering.ipynb`

5. **Prediction & Evaluation**  
   Run: `pred/pred.ipynb`

---

## 📁 Folder Structure

```plaintext
├── dataset/
│   ├── stages.csv
│   ├── races.csv
├── pred/
│   ├── pred.ipynb
├── 1data_transformation.ipynb
├── 2outlier_detection_pre.ipynb
├── 3aggregation.ipynb
├── 3aggregation_clustering.ipynb
├── clustering.ipynb
├── Cycling_Project.pdf
└── README.md
