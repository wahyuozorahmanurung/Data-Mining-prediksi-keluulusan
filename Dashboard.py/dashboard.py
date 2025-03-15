import plotly.express as px
import pandas as pd
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

sns.set(style='dark')

data_cleaned = pd.read_csv("data_cleaned.csv")

with st.sidebar:
    st.title("Prediksi Kelulusan")
    st.markdown("### 📊 Visualisasi Data Analisis dan Model")
    
    start_date = pd.to_datetime("2024-01-01")
    end_date = pd.to_datetime("2025-12-31")
    selected_date = st.date_input("📅 Pilih Tanggal", value=start_date, min_value=start_date, max_value=end_date)
    st.write("📌 Tanggal yang dipilih:", selected_date)
    
    selected_model = st.selectbox("📊 Pilih Model:", ["Decision Tree C4.5", "Random Forest", "SVM"])

# Simulasi hasil model
model_names = ["Decision Tree C4.5", "Random Forest", "SVM"]
accuracies = [0.9499, 0.9520, 0.9290]

st.subheader("📊 Perbandingan Akurasi Model")
fig = px.bar(x=model_names, y=accuracies, labels={'x': "Model", 'y': "Akurasi"},
             color=model_names, title="Perbandingan Akurasi Model", text_auto=".3f")
st.plotly_chart(fig)
st.write("Random Forest atau Decision Tree C4.5 lebih direkomendasikan untuk dataset ini karena memiliki prediksi yang lebih stabil dan lebih sedikit kesalahan dibandingkan SVM.")

# Confusion Matrix
cm_data = {
    "Decision Tree C4.5": np.array([[321, 4], [20, 134]]),
    "Random Forest": np.array([[321, 4], [19, 135]]),
    "SVM": np.array([[310, 15], [19, 135]])
}

st.subheader(f"📊 Confusion Matrix - {selected_model}")
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm_data[selected_model], annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Prediksi")
ax.set_ylabel("Aktual")
ax.set_title(f"Confusion Matrix - {selected_model}")
st.pyplot(fig)
st.write("Decision Tree C4.5 dan Random Forest memiliki performa yang mirip, dengan 321 TN, 4 FP, 19-20 FN, dan 134-135 TP. SVM memiliki sedikit lebih banyak kesalahan (15 FP), yang bisa menunjukkan kelemahan dalam memisahkan kelas negatif.")

# Distribusi Kelulusan
st.subheader("📊 Distribusi Kelulusan Mahasiswa")
fig = px.histogram(data_cleaned, x="GraduationStatus", color="GraduationStatus",
                   labels={'GraduationStatus': "Status Kelulusan"}, title="Distribusi Kelulusan Mahasiswa")
st.plotly_chart(fig)
st.write("Diagram ini menampilkan jumlah mahasiswa yang lulus dan tidak lulus berdasarkan data yang telah dianalisis.")

# Korelasi Fitur terhadap Kelulusan
st.subheader("📊 Korelasi Fitur terhadap Kelulusan")
correlations = data_cleaned.corr()["GraduationStatus"].drop("GraduationStatus").sort_values()
fig = px.bar(x=correlations.index, y=correlations.values, title="Korelasi Fitur terhadap Kelulusan",
             labels={'x': "Fitur", 'y': "Korelasi"}, color=correlations.values)
st.plotly_chart(fig)
st.write("GPA dan absensi adalah faktor paling berpengaruh dalam prediksi. Waktu belajar memiliki dampak sedang. Pendidikan orang tua, bimbingan, dan sukarela kurang signifikan.")

# Filter untuk Histogram
st.subheader("📊 Analisis Fitur Berdasarkan Kelulusan")

col1, col2 = st.columns(2)
with col1:
    gpa_range = st.slider("🎓 Pilih Rentang GPA", min_value=float(data_cleaned["GPA"].min()), 
                          max_value=float(data_cleaned["GPA"].max()), value=(data_cleaned["GPA"].min(), data_cleaned["GPA"].max()))
    filtered_data = data_cleaned[(data_cleaned["GPA"] >= gpa_range[0]) & (data_cleaned["GPA"] <= gpa_range[1])]
    fig = px.histogram(filtered_data, x="GPA", color="GraduationStatus", title="Distribusi GPA Berdasarkan Kelulusan")
    st.plotly_chart(fig)
    st.write(" 1. Mahasiswa dengan GPA lebih tinggi memiliki peluang lebih besar untuk lulus \n" 
             " 2. Sebagian besar mahasiswa yang tidak lulus memiliki GPA rendah, menunjukkan bahwa GPA merupakan faktor penting dalam keberhasilan akademik. \n"
             " \n Kesimpulan: Meningkatkan GPA melalui bimbingan belajar, mentoring, atau program peningkatan akademik bisa menjadi strategi efektif untuk meningkatkan tingkat kelulusan.")

with col2:
    absence_range = st.slider("📅 Pilih Rentang Absensi", 
                              min_value=int(data_cleaned["Absences"].min()), 
                              max_value=int(data_cleaned["Absences"].max()), 
                              value=(int(data_cleaned["Absences"].min()), int(data_cleaned["Absences"].max())), 
                              step=1)
    filtered_data = data_cleaned[(data_cleaned["Absences"] >= absence_range[0]) & (data_cleaned["Absences"] <= absence_range[1])]
    fig = px.histogram(filtered_data, x="Absences", color="GraduationStatus", title="Distribusi Kehadiran Berdasarkan Kelulusan")
    st.plotly_chart(fig)
    st.write(" 1. Mahasiswa yang memiliki banyak absensi cenderung memiliki tingkat kelulusan lebih rendah. \n"
"2. Sebagian besar mahasiswa yang lulus memiliki kehadiran yang tinggi, menunjukkan bahwa kehadiran dalam kelas berperan penting dalam keberhasilan akademik. \n"
"3.Jika ada mahasiswa yang banyak absen tapi tetap lulus, kemungkinan mereka memiliki cara belajar lain seperti belajar mandiri atau akses ke materi kuliah yang cukup \n"
"\n Menjaga disiplin kehadiran di kelas dan memastikan mahasiswa memiliki akses ke materi jika mereka terpaksa absen bisa membantu meningkatkan kelulusan.")

# Histogram Waktu Belajar
study_range = st.slider("📚 Pilih Rentang Waktu Belajar", 
                        min_value=float(data_cleaned["StudyTimeWeekly"].min()), 
                        max_value=float(data_cleaned["StudyTimeWeekly"].max()), 
                        value=(float(data_cleaned["StudyTimeWeekly"].min()), float(data_cleaned["StudyTimeWeekly"].max())), 
                        step=0.1)
filtered_data = data_cleaned[(data_cleaned["StudyTimeWeekly"] >= study_range[0]) & (data_cleaned["StudyTimeWeekly"] <= study_range[1])]
st.write("Histogram ini menampilkan waktu belajar mingguan mahasiswa dan dampaknya terhadap kelulusan.")
fig = px.histogram(filtered_data, x="StudyTimeWeekly", color="GraduationStatus", title="Distribusi Waktu Belajar per Minggu Berdasarkan Kelulusan")
st.plotly_chart(fig)

st.subheader("📊 Distribusi Waktu Studi per Minggu")
st.write("Grafik ini menunjukkan bagaimana distribusi waktu studi mahasiswa dalam seminggu.")
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(data_cleaned["StudyTimeWeekly"], bins=20, kde=True, color="blue", ax=ax)
ax.set_title("Distribusi Waktu Studi per Minggu")
ax.set_xlabel("Waktu Studi (jam/minggu)")
ax.set_ylabel("Jumlah Mahasiswa")
st.pyplot(fig)
