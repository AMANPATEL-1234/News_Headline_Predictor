import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
# Load model
model = tf.keras.models.load_model("news_model.h5")
# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)
# Use same max length from training
MAX_LEN = 200  
st.title("📰 News Category Classifier")
st.write("Enter a news headline/text and get predicted category.")
# Input text
text = st.text_area("Enter News Text:")
if st.button("Predict"):
    if text.strip():
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=MAX_LEN)
        pred = model.predict(padded)
        label = le.inverse_transform([np.argmax(pred)])
        st.success(f"Predicted Category: **{label[0]}**")
        K.clear_session()
    else:
        st.warning("Please enter some text.")

