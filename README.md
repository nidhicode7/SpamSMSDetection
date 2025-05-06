# SpamSMSDetection, Classifying SMS messages as spam or legitimate



# 📩 Spam SMS Detection

This project focuses on building a machine learning model to classify SMS messages as **Spam** or **Legitimate (Ham)**. It demonstrates a basic text classification pipeline including data preprocessing, feature extraction, and model training.

## 🔍 Overview

Unwanted spam messages are a major nuisance and can often be used in scams or phishing attacks. This project uses a labeled dataset of SMS messages to train and evaluate classification models that can distinguish between spam and non-spam texts.

## 🧠 Features

* Preprocessing of raw text (removing stop words, punctuation, etc.)
* Tokenization and vectorization using TF-IDF
* Training classification models (e.g., Naive Bayes, SVM, Logistic Regression)
* Evaluation using metrics like accuracy, precision, recall, and F1-score
* Visualization of word frequency in spam vs ham messages

## 📁 Dataset

* Dataset: SMS Spam Collection Dataset
* Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
* Format: CSV with two columns – `label` (ham/spam), `message` (text content)

## 🛠️ Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* NLTK or spaCy (for NLP preprocessing)
* Matplotlib, Seaborn

## 📊 Output

* Classification report and confusion matrix
* Word cloud for spam and ham messages
* Trained model to predict new SMS inputs

## 🚀 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/spam-sms-detection.git
   ```
2. Navigate to the project folder.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Run the notebook or Python script:

   ```bash
   python sms_classifier.py
   ```

## 🏁 Future Enhancements

* Deploy as a web app using Streamlit or Flask
* Real-time SMS prediction interface
* Incorporate deep learning methods (e.g., LSTM) for better accuracy

## 📜 License

This project is licensed under the MIT License.

