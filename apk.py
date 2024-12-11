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
        st.subheader("Hasil Preprocessing")
    
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


    elif selected == "Word2Vec":
        st.subheader("Hasil Word2Vec")
        
        import streamlit as st
        import pandas as pd
        import re
        import nltk
        from gensim.models import Word2Vec
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
        
        # Unduh stopwords bahasa Inggris (opsional jika diperlukan)
        nltk.download('stopwords')
        
        # Fungsi cleaning teks
        def cleaning(text):
            try:
                text = re.sub(r'\$\w*', '', text)  # Menghapus simbol $
                text = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))', '', text)  # URL
                text = re.sub(r'[^a-zA-Z\s]', '', text)  # Karakter non-alfabet
                text = re.sub(r'\b[a-zA-Z]{1,2}\b', '', text)  # Kata pendek 1-2 huruf
                text = re.sub(r'\s+', ' ', text).strip()  # Spasi berlebih
                return text
            except Exception as e:
                st.write(f"Error cleaning text: {e}")
                return text
        
        # Mengambil data
        df = pd.read_csv("https://raw.githubusercontent.com/citraaa12/skripsi/main/dataset.csv")
        df['komentar'] = df['komentar'].fillna("")
        
        # Preprocessing
        df['Cleaning'] = df['komentar'].apply(cleaning)
        df['CaseFolding'] = df['Cleaning'].str.lower()
        
        # Tokenizing
        df['Tokenizing'] = df['CaseFolding'].apply(lambda x: x.split())
        
        # Stopword Removal
        factory = StopWordRemoverFactory()
        stopword = factory.get_stop_words()
        
        def remove_stopwords(tokens):
            return [word for word in tokens if word not in stopword]
        
        df['stopword_removal'] = df['Tokenizing'].apply(remove_stopwords)
        
        # Pelatihan Word2Vec
        model_w2v = Word2Vec(
            sentences=df['stopword_removal'], 
            vector_size=100, 
            window=5, 
            min_count=1, 
            workers=4
        )
        
        # Ekstraksi fitur Word2Vec per kata
        def get_word_vectors(tokens):
            word_vectors = {}
            for word in tokens:
                if word in model_w2v.wv:
                    word_vectors[word] = model_w2v.wv[word]
            return word_vectors
        
        # Menambahkan kolom vektor per kata
        df['word2vec_per_word'] = df['stopword_removal'].apply(get_word_vectors)
        
        # Menampilkan hanya beberapa baris data
        st.dataframe(df.head(100))
        
        # Menyimpan model
        model_w2v.save('word2vec_model.model')
        st.write("Model Word2Vec telah disimpan.")

   
    elif selected == "Implementasi":
        st.subheader("Hasil Klasifikasi")

        import streamlit as st
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
        from sklearn.metrics import classification_report
        
        # Download NLTK stopwords
        nltk.download('stopwords')
        from nltk.corpus import stopwords
        
        # Stopword list
        stop_words = set(stopwords.words('indonesian'))
        
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
        st.write("Training LSTM model...")
        model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=1)
        
        # Streamlit App
        input_text = st.text_area("Masukkan komentar YouTube :")
        
        if st.button("Prediksi"):
            # Preprocess input text
            cleaned = clean_text(input_text)
            tokens = remove_stopwords(cleaned.split())
            sequence = tokenizer.texts_to_sequences([cleaned])
            padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
    
            # Predict sentiment
            prediction = model.predict(padded_sequence)[0][0]
            predicted_label = 1 if prediction > 0.5 else 0
    
            # Check if input exists in dataset
            original_label = "Tidak ditemukan dalam dataset"
            for i, row in df.iterrows():
                if row['cleaned'] == cleaned:
                    original_label = row['label']
                    break
    
            # Display results
            st.subheader("Hasil Prediksi")
            st.write(f"**LLabel Prediksi :** {predicted_label}")  # Changed to 0 or 1
            st.write(f"**Label Asli:** {original_label}")
        
        # Evaluate model
        st.subheader("Evaluasi Model")
        y_pred = (model.predict(X_test) > 0.5).astype("int32")
        st.write("Classification Report :")
        st.text(classification_report(y_test, y_pred))
            
st.markdown("---")  # Menambahkan garis pemisah
st.write("CITRA INDAH LESTARI - 200411100202 (TEKNIK INFORMATIKA)")
