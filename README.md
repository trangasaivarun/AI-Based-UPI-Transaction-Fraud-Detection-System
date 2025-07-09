# 🔐 UPI Fraud Detection System using Machine Learning

An end-to-end web-based machine learning project that detects **fraudulent UPI transactions**. This tool enables users to verify single or batch UPI transactions via a simple Streamlit interface and is backed by a highly accurate XGBoost classifier trained on real-world data.

---
![image](https://github.com/user-attachments/assets/edb6417d-ebb3-43ba-815d-0fa50dad5fbc)

---

## 📖 Introduction
Digital payments through UPI have revolutionized financial transactions in India. However, this growth has also increased the risk of fraud. This project presents a machine learning-based solution to detect UPI fraud using transactional patterns and behavior.

Built with a user-friendly Streamlit interface, it allows:
- 📥 Input of single or bulk transaction details
- 🤖 Real-time fraud detection
- 📈 Visual insights and downloadable results

The system integrates a well-optimized ML model with practical UI design to ensure ease of use, transparency, and security.

---

## 🌐 Live Demo
> 🔗 [🔍 Experience It Live](https://upi-fraud-detection-shwb6wmpvcbssp9dqumrs6.streamlit.app/)

---
## 📌 What’s Inside?

| File                                                                 | Description                                                                                 |
|----------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| `streamlit_app.py`                                                   | Interactive web app built with Streamlit for UPI fraud detection                           |
| `UPI Fraud Detection notebook.ipynb`                                 | Jupyter notebook for training and evaluating models like XGBoost and CatBoost              |
| `UPI Fraud Detection Final.pkl`                                      | Pre-trained XGBoost model used by the application (must be placed in the root directory)   |
| `Comparative_Analysis_of_UPI_Fraud_Detection_Using_Ensemble_Learning.pdf` | Published IEEE research paper explaining methodology and evaluation                        |

---

## 🎯 Project Goals
- **High Accuracy:** Achieve robust fraud detection using XGBoost  
- **Real-time Usability:** Enable immediate fraud prediction for users  
- **Accessible Interface:** Design a simple, intuitive web interface  
- **Research-backed Implementation:** Base the system on insights from ensemble learning methods, as published in an IEEE paper  

---

## 💻 Technologies Used

| Category         | Technology / Library         |
|------------------|------------------------------|
| ML Algorithms    |Logistic Regression, Decision Tree, Random Forest, Support Vector Machine, K-Nearest Neighbors, Naive Bayes, Gradient Boosting, XGBoost, LightGBM, CatBoost |
| Preprocessing    | SMOTE, One-Hot Encoding, Scaling |
| Backend          | Python, Streamlit            |
| Data Handling    | Pandas, NumPy                |
| Visualization    | Altair, Streamlit Metrics    |
| Model Deployment | Pickle                       |

---

## 📊 Dataset
The dataset contains over **50,000 UPI transactions** with features like:

- `amount`
- `transaction_type`
- `payment_gateway`
- `merchant_category`
- `transaction_state`
- `date` (used to extract `month` and `year`)

**Target:** Binary class — Fraudulent or Legitimate transaction

Preprocessing includes:
- One-hot encoding of categorical features
- SMOTE for class balancing
- Feature scaling using StandardScaler

---

## 🤖 Model Details
- **Model:** XGBoost Classifier  
- **Accuracy:** 99.53%  
- **Evaluation Metrics:**
  - Precision: 99.57%
  - Recall: 99.50%
  - F1-Score: 99.53%
  - ROC-AUC: 99.94%

Model evaluation was performed using hold-out validation and k-fold cross-validation.

---

## ⚙️ How It Works

### 🔹 Single Transaction Mode
1. Input transaction details manually
2. Model predicts if it’s fraudulent
3. Probability and transaction summary shown

### 🔹 Batch Mode
1. Upload a CSV file of transactions
2. System processes and predicts fraud
3. Displays insights and allows CSV download

---
### Workflow Diagram
![Block Diagram](https://github.com/user-attachments/assets/ba933ab8-24ef-44e4-a139-fe5b24eede85)


## 📸 Sample Outputs

| Legitimate Transaction | 
|------------------------|
| ![Legitimate]![Legitimate](https://github.com/user-attachments/assets/3a384433-5057-4e6f-8b77-6783f3d1312d)|

|Fraudulent Transaction |
------------------------|
| ![Fraudulent]![Fraud](https://github.com/user-attachments/assets/a1ba3cb7-609c-45e9-9a9a-b63bdf5a4a99)|

---

## 🧪 Evaluation Results

| Model                | Accuracy | Precision | Recall | ROC AUC |
|---------------------|----------|-----------|--------|---------|
| **XGBoost**         | 99.53%   | 99.57%    | 99.50% | 99.94%  |
| CatBoost            | 99.48%   | 99.46%    | 99.50% | 99.92%  |
| Logistic Regression | 99.23%   | 99.46%    | 98.99% | 99.82%  |

---

## 📄 Published Paper

**Title:** Comparative Analysis of UPI Fraud Detection Using Ensemble Learning  
**Conference:** 2025 International Conference on Computational Robotics, Testing and Engineering Evaluation (ICCRTEE)  
**DOI:** [10.1109/ICCRTEE64519.2025.11052942](https://ieeexplore.ieee.org/document/11052942)

---

## 🛠️ How to Run This Project Locally

Follow these steps to set up and run the UPI Fraud Detection project on your local machine.

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/trangasaivarun/UPI-Fraud-Detection.git
cd UPI-Fraud-Detection
```

### 2️⃣ Create and Activate Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Launch the Streamlit App

```bash
streamlit run streamlit_app.py
```

### 5️⃣ Access the App

After running the above command, open your browser and go to:

```
http://localhost:8501
```



