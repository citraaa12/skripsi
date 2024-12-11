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
        st.subheader("Hasil Klasifikasi")

        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
        
        # Fungsi untuk menampilkan hasil pengujian dan evaluasi
        def display_test_results(file_path):
            try:
                # Membaca data hasil testing
                results_df = pd.read_csv(file_path)
        
                # Tampilkan data hasil testing
                st.subheader("Hasil Testing")
                st.dataframe(results_df)
        
                # Hitung metrik evaluasi
                y_true = results_df['True Label']
                y_pred = results_df['Predicted Label']
        
                accuracy = accuracy_score(y_true, y_pred)
                st.write(f"**Akurasi:** {accuracy:.4f}")
        
                st.write("**Classification Report:**")
                report = classification_report(y_true, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())
        
                # Confusion Matrix
                cm = confusion_matrix(y_true, y_pred)
                st.write("**Confusion Matrix:**")
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted Label')
                ax.set_ylabel('True Label')
                ax.set_title('Confusion Matrix')
                st.pyplot(fig)
        
            except Exception as e:
                st.error(f"Gagal memproses file: {e}")
        
        # Pilih file hasil testing
        file_path = st.text_input("Masukkan path file hasil testing (CSV):")
        if file_path:
            display_test_results(file_path)
            
st.markdown("---")  # Menambahkan garis pemisah
st.write("CITRA INDAH LESTARI - 200411100202 (TEKNIK INFORMATIKA)")
