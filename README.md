# 📰 News_Headline_Predictor

A **web-based application** that predicts the **category of news headlines** using a trained deep learning model. ✨  

This project demonstrates **Natural Language Processing (NLP)** techniques, including:  
- 🔤 **Text Preprocessing** – Clean and prepare text data  
- ✂️ **Tokenization** – Convert text into sequences for the model  
- 📏 **Sequence Padding** – Ensure uniform input length for the neural network  
- 🌐 **Word Embeddings (GloVe)** – Use pre-trained embeddings to represent words
  
🧠 The model is built using **TensorFlow/Keras** and can classify headlines into multiple categories.  

---

## 💻 Features

- Predict the category of any news headline in real-time  
- Built with **Streamlit** for an interactive web interface  
- Uses pre-trained model (`news_model.h5`) and tokenizer (`tokenizer.pkl`)  

---

## 🛠️ Requirements

The app requires the following Python packages:

- tensorflow==2.19.0
- numpy==2.0.2
- scikit-learn==1.6.1
- pandas==2.2.2
- matplotlib==3.10.0
- seaborn==0.13.2
- nltk==3.9.1
- streamlit==1.37.0

⚠️ Make sure to use **Python 3.11** for TensorFlow compatibility.  

---

## 🚀 How to Run

1. Clone this repository:  
```bash
git clone https://github.com/AMANPATEL-1234/News_Headline_Predictor
```
2. Install dependencies:
pip install -r requirements.txt

3.Run the Streamlit app:
streamlit run app.py

---
📂 Repository Structure
```News_Headline_Predictor/
│── app.py
│── news_model.h5
│── tokenizer.pkl
│── label_encoder.pkl
│── requirements.txt
│── runtime.txt
│── README.md
```
---
## 📞 Contact
For any queries or collaboration, feel free to reach out:  
📱 **Phone:** +91-6392505818  
✉️ **Email:** amanpatel639250@gmail.com


