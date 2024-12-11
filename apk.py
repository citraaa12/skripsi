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
import nltk
nltk.download('stopwords')

import subprocess
try:
    import gensim
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gensim"])
    import gensim

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
        st.subheader("Word2Vec")

        # Load the dataset
        df = pd.read_csv("https://raw.githubusercontent.com/citraaa12/skripsi/main/preprocesing.csv")

        # Assume 'stopword_removal' is the column with the processed text for Word2Vec
        corpus = df['stopword_removal'].tolist()

        # Define parameters for Word2Vec
        DIM = 100  # Dimension of the word vectors
        corpus = [str(d) for d in corpus]  # Convert all elements in the corpus to strings
        tokenized_corpus = [d.split() for d in corpus]  # Tokenize text into words

        # Train Word2Vec model
        w2v_model = gensim.models.Word2Vec(sentences=tokenized_corpus, vector_size=DIM, window=10, min_count=1)

        # Extract vocabulary and vectors
        vocab = list(w2v_model.wv.index_to_key)
        vectors = [w2v_model.wv[word] for word in vocab]

        # Create a DataFrame to display words and their vectors
        vector_df = pd.DataFrame(vectors, index=vocab)

        # Display the vocabulary size
        st.write(f"Jumlah kata dalam vocab: {len(vocab)}")

        # Display the word vectors
        st.dataframe(vector_df, height=600, width=900)

        # Optional: Save the Word2Vec model for future use
        # w2v_model.save("word2vec_model.bin")
        # st.write("Model Word2Vec berhasil disimpan sebagai 'word2vec_model.bin'.")
    
    elif selected == "Information Gain":
        import requests
        from io import BytesIO
        st.subheader("Information Gain")
        st.write("Proses Information Gain")  # Debugging tambahan
        url = "https://raw.githubusercontent.com/dinia28/skripsi/main/hasil_ig.xlsx"
        response = requests.get(url)
        if response.status_code == 200:
            data = BytesIO(response.content)
            df = pd.read_excel(data)
            st.dataframe(df, width=600)
        else:
            st.error("Gagal mengambil file. Periksa URL atau koneksi internet.")
    
    elif selected == "Model WKNN":
        # Fungsi untuk memuat model dan menampilkan hasil rinci
        def load_and_display_model_details(percentage):
            model_filename = f"best_knn_model_{percentage}percent.pkl"
            results_filename = "training_results_with_rankings.xlsx"
            
            if not os.path.exists(model_filename):
                st.warning(f"Model untuk persentase {percentage}% tidak ditemukan.")
                return
        
            # Muat model
            best_model = joblib.load(model_filename)
            st.write(f"Model untuk {percentage}% dimuat.")
        
            # Muat hasil pelatihan dari file Excel
            if os.path.exists(results_filename):
                results = pd.read_excel(results_filename)
        
                # Filter hasil berdasarkan persentase fitur
                specific_results = results[results['Percentage'] == percentage]
                if specific_results.empty:
                    st.warning("Data hasil pelatihan tidak ditemukan untuk persentase ini.")
                    return
        
                # Menampilkan rincian hasil untuk setiap kombinasi parameter
                st.subheader("Detail Hasil untuk Kombinasi Parameter:")
                for index, row in specific_results.iterrows():
                    params = row['Best Parameters']
                    accuracy = row['Accuracy']
                    elapsed_time = row['Elapsed Time (s)']
                    st.write(f"Params: {params} | Accuracy: {accuracy:.4f} | Time: {elapsed_time:.2f} seconds")
                
                
                # Tampilkan informasi terbaik
                best_accuracy = specific_results['Accuracy'].max()
                best_params = specific_results.loc[specific_results['Accuracy'].idxmax(), 'Best Parameters']
                best_elapsed_time = specific_results.loc[specific_results['Accuracy'].idxmax(), 'Elapsed Time (s)']
                
                st.write(f"Model disimpan sebagai: {model_filename}")
                st.write(f"Best Params for {percentage}% features: {best_params}")
                st.write(f"Best Accuracy on Test Data: {best_accuracy:.4f}")
                st.write(f"Total Elapsed Time for Best Model: {best_elapsed_time:.2f} seconds")
            else:
                st.warning("File hasil pelatihan tidak ditemukan.")
        
        # Pilihan persentase yang dapat dipilih pengguna
        percentage_options = [95, 90, 85, 80, 75, 70, 65]
        selected_percentage = st.selectbox("Pilih Persentase Model :", percentage_options)
        
        # Memanggil fungsi untuk menampilkan detail model
        load_and_display_model_details(selected_percentage)
            
st.markdown("---")  # Menambahkan garis pemisah
st.write("CITRA INDAH LESTARI - 200411100202 (TEKNIK INFORMATIKA)")
