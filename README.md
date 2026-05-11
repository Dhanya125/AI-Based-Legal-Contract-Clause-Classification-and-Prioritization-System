# AI-Based Legal Contract Clause Classification and Prioritization System

An AI-powered legal contract analysis platform that automatically extracts, classifies, and prioritizes legal clauses from contract documents using **LegalBERT** and deep learning techniques.

The system helps legal professionals, businesses, and researchers reduce manual contract review effort while improving accuracy, scalability, and legal risk assessment.

---

## 📌 Overview

Legal contracts often contain hundreds of clauses that require detailed review and risk evaluation. Manual analysis is time-consuming, inconsistent, and difficult to scale.

This project provides an intelligent contract analysis solution that:

* Extracts clauses from uploaded PDF contracts
* Classifies clauses into predefined legal categories
* Assigns risk priority levels
* Predicts confidence scores
* Displays results through an interactive Flask dashboard

The model is built using **LegalBERT**, fine-tuned for multi-class legal clause classification.

---

# 🚀 Features

* 📄 PDF contract upload and processing
* 🔍 Automatic legal clause extraction
* 🧠 Clause classification using LegalBERT
* ⚠️ Risk prioritization:

  * High
  * Medium
  * Low
* 📊 Confidence score prediction
* 🌐 Flask-based interactive dashboard
* 📈 Summary statistics and clause insights
* 📤 Exportable analysis reports
* ⚡ Scalable legal contract intelligence pipeline

---

# 🛠️ Technologies Used

| Technology               | Purpose                      |
| ------------------------ | ---------------------------- |
| Python                   | Core development             |
| Flask                    | Web dashboard                |
| PyTorch                  | Deep learning framework      |
| HuggingFace Transformers | NLP model integration        |
| LegalBERT                | Legal language understanding |
| Scikit-learn             | Evaluation metrics           |
| PDF Processing Libraries | Text extraction              |
| HTML/CSS/Bootstrap       | Frontend interface           |

---

# 🏗️ System Architecture

```text
                +----------------------+
                | Upload Contract PDF |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | Text Extraction      |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | Clause Segmentation  |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | Text Preprocessing   |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | LegalBERT Classifier |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | Risk Prioritization  |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | Flask Dashboard      |
                +----------------------+
```

---

# ⚙️ Methodology

## 1️⃣ Data Preparation

* Dataset contains **30 labeled legal clause categories**
* Data split into:

  * Training set
  * Validation set
  * Testing set
* Label encoding converts textual labels into numerical values

---

## 2️⃣ Text Preprocessing

* LegalBERT tokenizer is used
* Sliding window tokenization handles long clauses
* Clauses converted into tensors using:

  * PyTorch Dataset
  * DataLoader

---

## 3️⃣ Model Training

* LegalBERT fine-tuned for multi-class classification
* Dropout regularization reduces overfitting
* AdamW optimizer used for training
* Learning rate scheduler improves convergence

---

## 4️⃣ Dashboard Integration

The Flask dashboard provides:

* Clause classification results
* Confidence scores
* Risk priority labels
* Contract summary statistics
* Interactive contract analysis workflow

---

# 📊 Model Performance

| Metric            | Training | Validation | Testing |
| ----------------- | -------- | ---------- | ------- |
| Loss              | 0.1579   | 0.8377     | -       |
| Accuracy          | 96.18%   | 78.28%     | 79.62%  |
| Macro F1 Score    | -        | 76.03%     | 78.31%  |
| Balanced Accuracy | -        | 76.29%     | 78.69%  |
| MCC               | -        | 77.21%     | 78.70%  |

---

# 📂 Legal Clause Categories

The model supports classification of clauses including:

* Confidentiality
* Indemnification
* Termination
* Payment Terms
* Liability
* Arbitration
* Intellectual Property
* Employment Terms
* Non-Disclosure
* Service Agreements
* Vendor Agreements
* And more...

---

# 💼 Applications

* Business contract analysis
* Employment agreement review
* Vendor contract validation
* Service agreement assessment
* Legal risk identification
* Automated legal intelligence systems

---

# ✅ Advantages

* Reduces manual legal review effort
* Faster contract processing
* Improved classification consistency
* Identifies high-risk legal clauses
* Efficient large-scale contract handling
* Enterprise-ready scalable architecture

---

# 🔮 Future Enhancements

* Multi-language legal contract support
* Named Entity Recognition (NER)
* Explainable AI for legal reasoning
* Real-time collaborative review
* Cloud deployment and API integration
* Advanced legal risk analytics

---

# 📁 Project Structure

```text
AI-Based-Legal-Contract-Clause-Classification/
│
├── app.py
├── model/
│   ├── best_model.pt
│   └── label_encoder.pkl
│
├── templates/
├── static/
├── uploads/
├── dataset/
├── notebooks/
├── requirements.txt
└── README.md
```

---

# ▶️ Installation

## 1. Clone Repository

```bash
git clone https://github.com/your-username/AI-Based-Legal-Contract-Clause-Classification.git
cd AI-Based-Legal-Contract-Clause-Classification
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Run Application

```bash
python app.py
```

---

## 4. Open in Browser

```text
http://127.0.0.1:5000/
```

---

# 📸 Dashboard Preview

Add screenshots of:

* Upload page
* Clause classification results
* Risk analysis dashboard
* Summary statistics

---

# 🤝 Contributors

* Dhanya S
* Project Team Members

---

# 📜 License

This project is developed for educational and research purposes.

---

# ⭐ Conclusion

This project demonstrates how transformer-based NLP models like LegalBERT can significantly improve legal contract analysis through automation, intelligent clause classification, and legal risk prioritization.

The system provides a scalable foundation for future AI-powered legal intelligence applications.
