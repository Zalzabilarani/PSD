import pickle
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

st.markdown(
    "<h1 style='text-align: center;'>Klasifikasi Diabetes Menggunakan Model Decision Tree Classifier</h1>", unsafe_allow_html=True
)
st.markdown(
    "<h4 style='text-align: center;'>Zalzabila Rani | 210411100082 | PSD - B</h4>", unsafe_allow_html=True
)

# st.info("Data diperoleh dari situs UCI Machine Learning dan dapat diakses pada link berikut : https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators ")

# load dataset -------------------------------------------------------------------
dataset = pd.read_csv('dataset_baru_dt.csv')

# split dataset menjadi data training dan data testing ---------------------------
fitur = dataset.drop(columns=['status'], axis =1)
target = dataset['status']
fitur_train, fitur_test, target_train, target_test = train_test_split(fitur, target, test_size = 0.2, random_state=42)

# normalisasi dataset ------------------------------------------------------------
# memanggil kembali model normalisasi zscore dari file pickle
with open('zcorescaler_baru.pkl', 'rb') as file_normalisasi:
    zscore = pickle.load(file_normalisasi)

zscoretraining = zscore.transform(fitur_train)
zscoretesting = zscore.transform(fitur_test)

# implementasi data pda model
with open('best_dt_model.pkl', 'rb') as file_model:
    model_dt = pickle.load(file_model)

model_dt.fit(zscoretraining, target_train)
prediksi_target = model_dt.predict(zscoretesting)


nodes = st.number_input ('Input nodes dari kelenjar getah bening anda.')

st.warning("Tekan 0 untuk 'tidak' dan 1 untuk 'ya'")

hormon = st.radio("Apakah Anda melakukan terapi?", ["none", "0", "1"])

jangka_waktu = st.number_input ('Input jangka waktu dari awal gejala hingga saat ini (hari).')

if st.button('Cek Status'):
    if nodes is not 0.0 and jangka_waktu is not "none" and hormon is not 0.0:
        prediksi = model_dt.predict([[nodes, jangka_waktu, hormon]])
        if prediksi== 0.0:
            st.success("Kondisi stabil (tanpa kekambuhan)!")
        elif prediksi == 1.0:
            st.error("Kekambuhan atau hasil yang kurang menguntungkan (termasuk kematian)")
    else:
        st.text('Data tidak boleh kosong. Harap isi semua kolom.')