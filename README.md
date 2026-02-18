# Credit Card Fraud Detection Model training with a neural network. 

## 🛡️ Problem Statement: Information Asymmetry in Finance
In the secondary financial market and digital banking, there is a significant **information asymmetry** between legitimate users and fraudulent actors. Traditional rule-based systems often fail to catch sophisticated patterns, and the extreme data imbalance (fraudulent transactions making up ~0.17% of the data) makes standard machine learning models biased toward the majority class.

Failure to detect these anomalies results in massive financial losses, while "false positives" (blocking a real customer) damage user trust and brand reputation.

## 🚀 The Solution: Deep Learning & Feature Importance
This project implements a **TensorFlow-based Deep Neural Network (DNN)** designed to mitigate this asymmetry. By analyzing hidden patterns in PCA-transformed transaction data, the model provides a transparent and automated tool for fair, real-time risk assessment.

### Key Impact:
* **Precision-Driven Detection:** Optimized to minimize false alarms while maintaining high recall for actual theft.
* **Scalable Architecture:** Built with TensorFlow, allowing for seamless deployment into production pipelines.
* **Feature Transparency:** Identifies which specific transaction behaviors (V1-V28 features) are the strongest indicators of risk.

---

## 🛠️ Technical Implementation

### Dataset
The model uses the [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset. It contains 284,807 transactions, where only 492 are fraudulent.

### Model Architecture
The network consists of a Sequential model with:
1. **Input Layer:** Normalized transaction features.
2. **Hidden Layers:** Multiple Dense layers with `ReLU` activation and `Dropout` (0.2) to prevent overfitting on the minority class.
3. **Output Layer:** `Sigmoid` activation to output a probability score between 0 and 1.

### Optimization Strategy
To handle the imbalance, the project utilizes:
* **Class Weighting:** Adjusting the loss function to penalize misclassifications of fraud more heavily.
* **Precision-Recall Metrics:** Focusing on AUPRC rather than Accuracy.


## 📋 Requirements
```bash
pip install tensorflow pandas scikit-learn matplotlib
