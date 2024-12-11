import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from numpy import array
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
import subprocess
subprocess.run(['pip', 'install', 'imbalanced-learn'])
from imblearn.over_sampling import RandomOverSampler
from math import sqrt
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
import re
import time
import seaborn as sns
import os
os.system('pip install nltk')
from nltk.stem import PorterStemmer


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

with st.container():
    with st.sidebar:
        selected = option_menu(
            st.write(
                """<h2 style = "text-align: center;"><img src="https://raw.githubusercontent.com/dinia28/skripsi/main/home.png" width="130" height="130"><br></h2>""",
                unsafe_allow_html=True,
            ),
            [
                "Home",
                "Data",
                "Preprocessing",
                "Word2Vec",
                "Implementasi",

            ],
            icons=[
                "house",
                "person",
                "gear",
                "bar-chart",
                "file-earmark-font",
            ],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#87CEEB"},
                "icon": {"color": "white", "font-size": "18px"},
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "left",
                    "margin": "0px",
                    "color": "white",
                },
                "nav-link-selected": {"background-color": "#005980"},
            },
        )

    if selected == "Home":
        st.write(
            """<h3 style = "text-align: center;">
        <img src="https://raw.githubusercontent.com/citraaa12/skripsi/main/ikn.png" width="500" height="300">
        </h3>""",
            unsafe_allow_html=True,
        )

        st.subheader("""Deskripsi Aplikasi""")
        st.write(
            """
        ANALISIS SENTIMEN PEMBANGUNAN IBUKOTA NUSANTARA MENGGUNAKAN METODE LONG SHORT TERM-MEMORY DAN WORD2VEC
        """
        )

    elif selected == "Data":

        st.subheader("""Deskripsi Data""")
        st.write(
            """
        Data yang digunakan dalam aplikasi ini yaitu data dari hasil crawling komentar pada video youtube
        """
        )
        
        st.subheader("Dataset")
        # Menggunakan file Excel dari GitHub
        df = pd.read_csv(
            "https://raw.githubusercontent.com/citraaa12/skripsi/main/dataset.csv"
        )
        st.dataframe(df, width=600)
        
        st.subheader("label")
        # Menampilkan frekuensi dari masing-masing label
        label_counts = df['label'].value_counts()
        st.write(label_counts)
        
    elif selected == "Preprocessing":
        # Cleansing
        st.subheader("Preprocessing")
    
        import streamlit as st
        import pandas as pd
        import re
        import nltk
        import gensim
        import numpy as np
        from gensim.models import Word2Vec
        from nltk.tokenize import word_tokenize
        
        # Unduh stopwords untuk bahasa Indonesia
        nltk.download('stopwords')
        
        # Fungsi untuk cleaning teks
        def cleaning(text):
            try:
                text = re.sub(r'\$\w*', '', text)  # Menghapus simbol atau kata yang dimulai dengan $
                text = re.sub(r'^rt[\s]+', '', text)  # Menghapus retweet mark
                text = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))', '', text)  # Menghapus URL
                text = re.sub('&quot;', " ", text)  # Menghapus tanda kutip
                text = re.sub(r"\d+", "", text)  # Menghapus angka
                text = re.sub(r"\b[a-zA-Z]\b", "", text)  # Menghapus huruf tunggal
                text = re.sub(r'[^\w\s]', '', text)  # Menghapus karakter selain kata
                text = re.sub(r'(.)\1+', r'\1\1', text)  # Mengurangi huruf berulang
                text = re.sub(r'\s+', ' ', text).strip()  # Menghapus spasi berlebih
                text = re.sub(r'#', '', text)  # Menghapus simbol hash
                text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Menghapus simbol non-alfabet
                text = re.sub(r'\b\w{1,2}\b', '', text)  # Menghapus kata dengan panjang 1-2 karakter
                text = re.sub(r'\s\s+', ' ', text).strip()  # Menghapus spasi ganda
                text = re.sub(r'^RT[\s]+', '', text)  # Menghapus indikasi retweet
                text = re.sub(r'^b[\s]+', '', text)  # Menghapus b
                text = re.sub(r'^link[\s]+', '', text)  # Menghapus link
                return text
            except Exception as e:
                st.write(f"Error cleaning text : {e}")
                return text
        
        # Mengambil data dari file CSV
        df = pd.read_csv("https://raw.githubusercontent.com/citraaa12/skripsi/main/dataset.csv")
        # Menampilkan data sebelum dibersihkan
        st.write("Data contoh sebelum cleaning:", df['komentar'].head())
        
        # Mengisi nilai NaN dengan string kosong
        df['komentar'] = df['komentar'].fillna("")
        
        # Menerapkan fungsi cleaning pada kolom 'komentar'
        df['Cleaning'] = df['komentar'].apply(cleaning)
        st.write("Hasil Cleaning:")
        st.dataframe(df[['komentar', 'Cleaning']])
        
        # Menambahkan proses Case Folding (huruf kecil)
        df['CaseFolding'] = df['Cleaning'].str.lower()
        st.write("Hasil Case Folding:")
        st.dataframe(df[['komentar', 'Cleaning', 'CaseFolding']])
        
        # Tokenizing
        def tokenizer(text):
            if isinstance(text, str):
                return text.split()  # Tokenisasi sederhana dengan split
            return []
        
        df['Tokenizing'] = df['CaseFolding'].apply(tokenizer)
        st.write("Hasil Tokenizing:")
        st.dataframe(df[['komentar', 'Cleaning', 'CaseFolding', 'Tokenizing']])
        
        # Stopword Removal
        stopword = nltk.corpus.stopwords.words('indonesian')
        
        def remove_stopwords(text):
            return [word for word in text if word not in stopword]
        
        df['stopword_removal'] = df['Tokenizing'].apply(lambda x: remove_stopwords(x))
        
        # Menampilkan hasil stopword removal
        st.write("Data setelah stopword removal:")
        st.dataframe(df[['komentar', 'Cleaning', 'CaseFolding', 'Tokenizing', 'stopword_removal']])
        
        # Fungsi untuk mengonversi list kata menjadi string
        def fit_stopwords(text):
            text = np.array(text)
            text = ' '.join(text)
            return text
        
        df['stopword_removal'] = df['stopword_removal'].apply(lambda x: fit_stopwords(x))
        
        # Menampilkan hasil akhir setelah stopword removal
        st.write("Data setelah stopword removal (dalam format teks):")
        st.dataframe(df[['komentar', 'Cleaning', 'CaseFolding', 'Tokenizing', 'stopword_removal']])
        
        # Tokenisasi setelah stopword removal
        tokenized_data = df['stopword_removal'].apply(lambda x: x.split()).tolist()
        
        # Melatih model Word2Vec
        model_w2v = Word2Vec(sentences=tokenized_data, vector_size=100, window=5, min_count=1, workers=4)
        
        # Fungsi untuk mendapatkan vektor representasi dari sebuah kalimat
        def get_vector(text):
            vector = np.zeros(100)  # Sesuaikan dengan ukuran vektor (100)
            words = text.split()
            count = 0
            for word in words:
                if word in model_w2v.wv:
                    vector += model_w2v.wv[word]  # Menambahkan vektor kata ke dalam total vektor kalimat
                    count += 1
            if count > 0:
                vector /= count  # Normalisasi dengan jumlah kata yang ada dalam kalimat
            return vector
        
        # Ekstraksi fitur menggunakan Word2Vec
        df['word2vec_features'] = df['stopword_removal'].apply(lambda x: get_vector(x))
        
        # Menampilkan data akhir dengan fitur Word2Vec
        st.write("Data dengan Fitur Word2Vec:")
        st.dataframe(df[['komentar', 'Cleaning', 'CaseFolding', 'Tokenizing', 'stopword_removal', 'word2vec_features']])
        
        # Menyimpan model Word2Vec untuk digunakan nanti
        model_w2v.save('word2vec_model.model')
        st.write("Model Word2Vec telah disimpan.")
        
        # Anda bisa menggunakan model ini untuk ekstraksi fitur lebih lanjut


    elif selected == "Word2Vec":
        st.subheader("Hasil Word2Vec")
        # Menggunakan file Excel dari GitHub
        df = pd.read_excel(
            "https://raw.githubusercontent.com/citraaa12/skripsi/main/w2v.xlsx"
        )
        st.dataframe(df, width=600)
    
    elif selected == "Implementasi":
        st.subheader("Hasil Klasifikasi")

        import streamlit as st
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
        import numpy as np
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        import nltk
        from nltk.corpus import stopwords
        import re
        
        # Unduh stopwords jika belum ada
        nltk.download('stopwords')
        stop_words = set(stopwords.words('indonesian'))
        
        # Parameter preprocessing
        max_length = 20  # Panjang input
        vocab_size = 14545  # Disesuaikan dengan model Anda
        
        def clean_text(text):
            text = re.sub(r'\$\w*', '', text)
            text = re.sub(r'^rt[\s]+', '', text)
            text = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))', '', text)
            text = re.sub('&quot;', " ", text)
            text = re.sub(r"\d+", "", text)
            text = re.sub(r"\b[a-zA-Z]\b", "", text)
            text = re.sub(r'[^\w\s]', '', text)
            text = re.sub(r'(.)\1+', r'\1\1', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        
        def case_folding(text):
            return text.lower()
        
        def tokenize(text):
            return text.split()
        
        def remove_stopwords(tokens):
            return [word for word in tokens if word not in stop_words]
        
        def preprocess_text(text):
            cleaned = clean_text(text)
            folded = case_folding(cleaned)
            tokens = tokenize(folded)
            no_stopwords = remove_stopwords(tokens)
            return " ".join(no_stopwords), no_stopwords
        
        def predict_sentiment(text):
            # Tokenizer
            tokenizer = Tokenizer(num_words=vocab_size)
            tokenizer.fit_on_texts([text])
            sequences = tokenizer.texts_to_sequences([text])
            padded = pad_sequences(sequences, maxlen=max_length, padding='post')
            prediction = model.predict(padded)
            return 1 if prediction[0][0] > 0.5 else 0, prediction[0][0]
        
        st.title("Analisis Sentimen Komentar YouTube")
        
        if model:
            # Input komentar
            input_text = st.text_area("Masukkan komentar YouTube:")
            if st.button("Analisis"):
                # Preprocessing
                cleaned, tokens = preprocess_text(input_text)
                st.subheader("Hasil Preprocessing")
                st.write(f"**Original Text:** {input_text}")
                st.write(f"**Cleaned Text:** {cleaned}")
                st.write(f"**Tokens:** {tokens}")
        
                # Predict sentiment
                label, score = predict_sentiment(cleaned)
                sentiment = "Positif" if label == 1 else "Negatif"
                st.subheader("Hasil Prediksi")
                st.write(f"**Sentimen:** {sentiment}")
                st.write(f"**Confidence Score:** {score:.4f}")
        
                # Confusion Matrix placeholder
                st.write("**Confusion Matrix:**")
                st.write("Data confusion matrix akan tersedia setelah evaluasi batch.")
        
                # TODO: Tambahkan evaluasi batch jika diperlukan
        else:
            st.error("Model tidak dapat dimuat. Silakan cek kembali.")

            
st.markdown("---")  # Menambahkan garis pemisah
st.write("CITRA INDAH LESTARI - 200411100202 (TEKNIK INFORMATIKA)")
