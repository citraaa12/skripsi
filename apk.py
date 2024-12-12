import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import re
import nltk
from gensim.models import Word2Vec
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

# Download NLTK stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Stopword list
stop_words = set(stopwords.words('indonesian'))

# Streamlit app configuration
st.set_page_config(
    page_title="Analisis Sentimen Pembangunan Ibukota Nusantara",
    page_icon="https://raw.githubusercontent.com/dinia28/skripsi/main/rumah.jpg",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://www.extremelycoolapp.com/help",
        "Report a bug": "https://www.extremelycoolapp.com/bug",
        "About": "# This is a header. This is an *extremely* cool app!",
    },
)

st.write(
    """<h1 style="font-size: 40px;">Analisis Sentimen Pembangunan Ibukota Nusantara</h1>""",
    unsafe_allow_html=True,
)

# Text preprocessing functions
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]

# Load dataset
df = pd.read_csv("https://raw.githubusercontent.com/citraaa12/skripsi/main/dataset.csv")
df['komentar'] = df['komentar'].fillna("")

# Preprocessing
df['cleaned'] = df['komentar'].apply(clean_text)
df['tokens'] = df['cleaned'].apply(lambda x: x.split())
df['tokens'] = df['tokens'].apply(remove_stopwords)

# Train Word2Vec model
model_w2v = Word2Vec(sentences=df['tokens'], vector_size=100, window=5, min_count=1, workers=4)

# Convert tokens to Word2Vec average vectors
def vectorize(tokens):
    vectors = [model_w2v.wv[word] for word in tokens if word in model_w2v.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model_w2v.vector_size)

df['vectorized'] = df['tokens'].apply(vectorize)

# Prepare data for LSTM
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['cleaned'])
sequences = tokenizer.texts_to_sequences(df['cleaned'])
vocab_size = len(tokenizer.word_index) + 1

max_length = 20
X = pad_sequences(sequences, maxlen=max_length, padding='post')
y = df['label'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build LSTM model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=100, input_length=max_length),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=1)

# Streamlit sidebar menu
with st.sidebar:
    selected = option_menu(
        "Menu",
        ["Home", "Data", "Preprocessing", "Word2Vec", "Implementasi"],
        icons=["house", "database", "gear", "diagram-3", "check2-circle"],
        menu_icon="cast",
        default_index=0,
    )

if selected == "Home":
    st.write(
        """<h3 style = "text-align: center;">
        <img src="https://raw.githubusercontent.com/citraaa12/skripsi/main/ikn.png" width="500" height="300">
        </h3>""",
        unsafe_allow_html=True,
    )
    st.subheader("Deskripsi Aplikasi")
    st.write("ANALISIS SENTIMEN PEMBANGUNAN IBUKOTA NUSANTARA MENGGUNAKAN METODE LONG SHORT TERM-MEMORY DAN WORD2VEC")

elif selected == "Data":
    st.subheader("Deskripsi Data")
    st.write("Data yang digunakan dalam aplikasi ini yaitu data dari hasil crawling komentar pada video YouTube")
    st.dataframe(df[['komentar', 'label']])

elif selected == "Preprocessing":
    st.subheader("Hasil Preprocessing")
    st.dataframe(df[['komentar', 'cleaned', 'tokens']])

elif selected == "Word2Vec":
    st.subheader("Hasil Word2Vec")
    st.write("Contoh vektor Word2Vec untuk setiap komentar:")
    st.dataframe(df[['komentar', 'vectorized']].head())

elif selected == "Implementasi":
    st.subheader("Hasil Klasifikasi")

    input_text = st.text_area("Masukkan data komentar YouTube :")

    if st.button("Prediksi"):
        # Preprocess input text
        cleaned = clean_text(input_text)
        tokens = remove_stopwords(cleaned.split())
        sequence = tokenizer.texts_to_sequences([cleaned])
        padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')

        # Predict sentiment
        prediction = model.predict(padded_sequence)[0][0]
        predicted_label = 1 if prediction > 0.5 else 0

        # Display results
        st.subheader("Hasil Prediksi")
        if predicted_label == 1:
            st.markdown(
                """
                <div style="background-color:#d4edda; color:#155724; padding:10px; border:2px solid #c3e6cb; border-radius:5px; text-align:center;">
                    <strong>Positif</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="background-color:#f8d7da; color:#721c24; padding:10px; border:2px solid #f5c6cb; border-radius:5px; text-align:center;">
                    <strong>Negatif</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.markdown("---")
st.write("CITRA INDAH LESTARI - 200411100202 (TEKNIK INFORMATIKA)")
