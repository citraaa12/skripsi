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
    
        import re
        import pandas as pd
        
        # Mendefinisikan fungsi cleaning
        def cleaning(text):
            try:
                text = re.sub(r'\$\w*', '', text)
                text = re.sub(r'^rt[\s]+', '', text)
                text = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))', '', text)
                text = re.sub('&quot;', " ", text)
                text = re.sub(r"\d+", "", text)
                text = re.sub(r"\b[a-zA-Z]\b", "", text)
                text = re.sub(r'[^\w\s]', '', text)
                text = re.sub(r'(.)\1+', r'\1\1', text)
                text = re.sub(r'\s+', ' ', text).strip()
                text = re.sub(r'#', '', text)
                text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
                text = re.sub(r'\b\w{1,2}\b', '', text)
                text = re.sub(r'\s\s+', ' ', text).strip()
                text = re.sub(r'^RT[\s]+', '', text)
                text = re.sub(r'^b[\s]+', '', text)
                text = re.sub(r'^link[\s]+', '', text)
                return text
            except Exception as e:
                st.write(f"Error cleaning text : {e}")
                return text
        
        # Mengambil data dari file csv
        df = pd.read_csv("https://raw.githubusercontent.com/citraaa12/skripsi/main/dataset.csv")
        # Cek kolom dan isi untuk memastikan kolom 'Ulasan' ada
        st.write("Data contoh sebelum cleaning :", df['komentar'].head())
        
        # Mengisi nilai NaN dengan string kosong untuk kolom 'Ulasan'
        df['komentar'] = df['komentar'].fillna("")
        
        # Menerapkan fungsi cleaning
        df['Cleaning'] = df['komentar'].apply(cleaning)
        st.write("Hasil Cleaning :")
        st.dataframe(df[['komentar', 'Cleaning']])
        
        # Menambahkan proses case folding
        df['CaseFolding'] = df['Cleaning'].str.lower()
        st.write("Hasil Case Folding :")
        st.dataframe(df[['komentar', 'Cleaning', 'CaseFolding']])

        # Tokenizing
        def tokenizer(text):
            if isinstance(text, str):
                return text.split()  # Tokenisasi sederhana dengan split
                return []
        
        # Menerapkan tokenizing pada kolom 'case folding'
        df['Tokenizing'] = df['CaseFolding'].apply(tokenizer)
        
        # Tampilkan hasil akhir setelah tokenizing
        st.write("Hasil Tokenizing :")
        st.dataframe(df[['komentar', 'Cleaning', 'CaseFolding', 'Tokenizing']])
        
        # Stopword removal
        import pandas as pd
        import nltk
        from nltk.tokenize import word_tokenize
        import numpy as np

        # Unduh package stopwords untuk bahasa Indonesia
        nltk.download('stopwords')

        # Stopword Removal
        stopword = nltk.corpus.stopwords.words('indonesian')

        def remove_stopwords(text):
            return [word for word in text if word not in stopword]

        df['stopword_removal'] = df['Tokenizing'].apply(lambda x: remove_stopwords(x))
        df.head()

        # Remove karakter
        stopword_removal = df[['stopword_removal']]

        def fit_stopwords(text):
            text = np.array(text)
            text = ' '.join(text)
            return text

        df['stopword_removal'] = df['stopword_removal'].apply(lambda x: fit_stopwords(x))
        
        # Menampilkan hasil di Streamlit
        st.write("Data setelah stopword removal :")
        st.dataframe(df[['komentar', 'Cleaning', 'CaseFolding', 'Tokenizing', 'stopword_removal']])

    elif selected == "Word2Vec":
        st.subheader("Hasil Word2Vec")
        # Menggunakan file Excel dari GitHub
        df = pd.read_excel(
            "https://raw.githubusercontent.com/citraaa12/skripsi/main/w2v.xlsx"
        )
        st.dataframe(df, width=600)
    
    elif selected == "Implementasi":
        st.subheader("Implementasi Model")

        # Fungsi untuk memuat model yang telah dilatih
        @st.cache_resource
        def load_model():
            # Load model terlatih
            model_path = "best_lstm_model.h5"  # Sesuaikan dengan path model Anda
            model = keras.models.load_model(model_path)
            return model
    
        # Memuat model
        model = load_model()
        st.write("Model berhasil dimuat.")
    
        # Memasukkan komentar untuk prediksi
        komentar_input = st.text_area("Masukkan komentar YouTube untuk diprediksi :")
    
        if st.button("Prediksi"):
            if komentar_input.strip():
                # Preprocessing komentar
                def preprocess_text(text):
                    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
                    text = text.lower()
                    return text
    
                komentar_preprocessed = preprocess_text(komentar_input)
                st.write("Komentar setelah preprocessing :", komentar_preprocessed)
    
                # Tokenisasi dan padding
                tokenizer = joblib.load("tokenizer.pkl")  # Pastikan tokenizer disimpan saat melatih model
                sequences = tokenizer.texts_to_sequences([komentar_preprocessed])
                padded_sequences = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=20)
    
                # Prediksi
                prediction = model.predict(padded_sequences)
                label_prediksi = "Positif" if prediction[0][0] > 0.5 else "Negatif"
    
                st.write(f"Hasil Prediksi: {label_prediksi}")
    
                # Menampilkan label sebenarnya jika tersedia
                true_label = st.selectbox("Pilih label sebenarnya (opsional) :", ["", "Positif", "Negatif"])
                if true_label:
                    st.write(f"Label Sebenarnya: {true_label}")
    
            else:
                st.warning("Harap masukkan komentar untuk diprediksi.")
    
        # Evaluasi Model
        st.subheader("Evaluasi Model")
    
        # Memuat data uji dan hasil prediksi
        if st.button("Tampilkan Evaluasi"):
            try:
                # Data uji (pastikan ini sesuai dengan data yang digunakan untuk evaluasi)
                x_test = np.load("x_test.npy")
                y_test = np.load("y_test.npy")
    
                # Prediksi data uji
                y_pred = model.predict(x_test)
                y_pred_classes = (y_pred > 0.5).astype(int)
    
                # Evaluasi metrik
                accuracy = accuracy_score(y_test, y_pred_classes)
                cm = confusion_matrix(y_test, y_pred_classes)
                report = classification_report(y_test, y_pred_classes, output_dict=True)
    
                st.write(f"Akurasi : {accuracy:.4f}")
    
                # Tampilkan confusion matrix
                st.write("Confusion Matrix:")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
                st.pyplot(fig)
    
                # Tampilkan classification report
                st.write("Classification Report:")
                st.json(report)
    
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
            
st.markdown("---")  # Menambahkan garis pemisah
st.write("CITRA INDAH LESTARI - 200411100202 (TEKNIK INFORMATIKA)")
