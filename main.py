import pickle
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

st.markdown(
    "<h1 style='text-align: center;'>Klasifikasi Kanker Payudara Menggunakan Model Decision Tree Classifier</h1>", unsafe_allow_html=True
)
st.markdown(
    "<h4 style='text-align: center;'>Zalzabila Rani | 210411100082 | PSD - B</h4>", unsafe_allow_html=True
)

# load dataset -------------------------------------------------------------------
dataset = pd.read_csv('dataset_baru.csv')

# split dataset menjadi data training dan data testing ---------------------------
fitur = dataset.drop(columns=['status'], axis =1)
target = dataset['status']
fitur_train, fitur_test, target_train, target_test = train_test_split(fitur, target, test_size = 0.2, random_state=42)

# normalisasi dataset ------------------------------------------------------------
# memanggil kembali model normalisasi zscore dari file pickle
with open('zscore_scaler.pkl', 'rb') as file_normalisasi:
    zscore = pickle.load(file_normalisasi)

zscoretraining = zscore.transform(fitur_train)
zscoretesting = zscore.transform(fitur_test)

# implementasi data pda model
with open('model_nb.pkl', 'rb') as file_model:
    model_dt = pickle.load(file_model)

model_dt.fit(zscoretraining, target_train)
prediksi_target = model_dt.predict(zscoretesting)


age = st.number_input ('Input usia Anda.')

size = st.number_input ('Input ukuran tumor Anda dalam milimeter.')

grade = st.radio("Pilih tingkat keparahan tumor:", ["none", "1", "2", "3"])

nodes = st.number_input ('Input nodes dari kelenjar getah bening Anda.')

pgr = st.text_input('Input kadar reseptor progesteron dalam darah Anda.', '0')

st.warning("Tekan 0 untuk 'tidak' dan 1 untuk 'ya'")

hormon = st.radio("Apakah Anda melakukan terapi?", ["none", "0", "1"])

jangka_waktu = st.number_input('Input jangka waktu dari awal gejala hingga saat ini (hari).')

# if st.button('Cek Status'):
#     if age != 0.0 and size != 0.0 and grade is not None and nodes != 0.0 and pgr != 0.0 and jangka_waktu is not None and hormon is not None:
#         prediksi = model_dt.predict([[age, size, grade, nodes, pgr, jangka_waktu, hormon]])
#         if prediksi == 0.0:
#             st.success("Kondisi stabil (tanpa kekambuhan)!")
#         elif prediksi == 1.0:
#             st.error("Kekambuhan atau hasil yang kurang menguntungkan (termasuk kematian)")
#     else:
#         st.text('Data tidak boleh kosong. Harap isi semua kolom.')

# Prediksi
if st.button('Cek Status'):
    if all([age, size, grade, nodes, pgr, jangka_waktu, hormon]):
        input_data = [[age, size, int(grade.split()[-1]), nodes, pgr, jangka_waktu, int(hormon.split()[-1])]]
        prediksi = model_dt.predict(zscore.transform(input_data))
        
        if prediksi == 0:
            st.success("Kondisi stabil (tanpa kekambuhan)!")
        elif prediksi == 1:
            st.error("Kekambuhan atau hasil yang kurang menguntungkan (termasuk kematian)")
    else:
        st.text('Data tidak boleh kosong. Harap isi semua kolom.')