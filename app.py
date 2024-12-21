from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
from dotenv import load_dotenv
print("dotenv imported successfully")

# Memuat variabel lingkungan dari file .env
load_dotenv()

# Membuat aplikasi Flask
app = Flask(__name__)

# Menggunakan SECRET_KEY yang dimuat dari .env
app.secret_key = os.getenv('SECRET_KEY')

# Cek apakah secret_key berhasil dimuat
print("Secret Key:", app.secret_key)  # Optional, untuk memverifikasi secret_key

# Fungsi untuk memuat model
def load_new_model():
    model_path = 'model/stress_level_model_v2.h5'  # Perbaiki path
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None  # Kembalikan None jika gagal

# Fungsi untuk memuat dataset dan scaler
def load_dataset_and_scaler():
    dataset_path = 'StressLevelDataset.csv'  # Perbaiki path
    try:
        df = pd.read_csv(dataset_path)
        print(f"Dataset loaded successfully from {dataset_path}")
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df.drop(columns=['stress_level']))
        return df, scaler
    except FileNotFoundError:
        print("File dataset tidak ditemukan!")
        return None, None

# Fungsi untuk memprediksi tingkat stres
def predict_stress(model, scaler, user_input):
    if model is None or scaler is None:
        return -1 
    user_input_scaled = scaler.transform([user_input])
    prediction = model.predict(user_input_scaled)
    predicted_class = np.argmax(prediction, axis=1)
    return predicted_class[0]

# Fungsi untuk mendapatkan pertanyaan dan min-max
def get_stress_questions_and_min_max():
    df = pd.read_csv('/Users/macrizkyjackbar/Pribadi/Skripsi/stresslevel/StressLevelDataset.csv')  # Perbaiki path
    questions = {
        'anxiety_level': "Akhir-akhir ini, sering nggak ngerasa cemas atau gelisah?",
        'self_esteem': "Gimana nih, kamu lagi pede sama diri sendiri atau nggak?",
        'mental_health_history': "Pernah punya masalah kesehatan mental sebelumnya?",
        'depression': "Belakangan ini, sering ngerasa sedih atau tertekan nggak?",
        'headache': "Sakit kepala atau migrain, sering nggak akhir-akhir ini?",
        'blood_pressure': "Gimana kondisi tekanan darah kamu belakangan ini?",
        'sleep_quality': "Tidur kamu nyenyak nggak belakangan ini?",
        'breathing_problem': "Ada masalah nafas atau sering sesak nggak?",
        'noise_level': "Lingkungan kamu berisik nggak? Kayak di rumah, kampus, atau tempat kerja.",
        'living_conditions': "Menurut kamu, kondisi tempat tinggal kamu gimana?",
        'safety': "Kamu ngerasa aman nggak di lingkungan tempat tinggal kamu?",
        'basic_needs': "Kebutuhan pokok kamu, kayak makan, tempat tinggal, udah cukup belum?",
        'academic_performance': "Menurut kamu, gimana nilai atau performa akademis kamu?",
        'study_load': "Beban tugas atau belajar kamu akhir-akhir ini berat nggak?",
        'teacher_student_relationship': "Hubungan kamu sama guru atau dosen gimana?",
        'future_career_concerns': "Khawatir nggak sama masa depan atau karier kamu?",
        'social_support': "Kamu ngerasa didukung sama keluarga atau teman nggak?",
        'peer_pressure': "Kamu sering ngerasa ditekan sama teman sebaya nggak?",
        'extracurricular_activities': "Kamu sering ikut kegiatan ekstrakurikuler atau kegiatan lain nggak?",
        'bullying': "Pernah nggak ngalamin bullying di sekolah/kampus atau di tempat lain ?"
    }
    min_max_values = {col: (df[col].min(), df[col].max()) for col in df.columns if col != 'stress_level'}
    return questions, min_max_values

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'start' in request.form:
            questions, _ = get_stress_questions_and_min_max()
            session['current_index'] = 0
            session['user_input'] = [None] * len(questions)
            return redirect(url_for('questions'))

    return render_template('index.html')

@app.route("/questions", methods=["GET", "POST"])
def questions():
    questions, min_max_values = get_stress_questions_and_min_max()
    question_keys = list(questions.keys())
    current_index = session.get('current_index', 0)
    user_input = session.get('user_input', [None] * len(question_keys))

    # Validasi current_index agar tidak melampaui batas
    if current_index < 0 or current_index >= len(question_keys):
        return redirect(url_for('index'))

    if request.method == "POST":
        if 'next' in request.form:
            value = request.form.get('answer', None)
            if value is not None:
                user_input[current_index] = float(value)
            current_index += 1
        elif 'back' in request.form:
            current_index -= 1
        elif 'predict' in request.form:
            value = request.form.get('answer', None)
            if value is not None:
                user_input[current_index] = float(value)

            # Prediksi tingkat stres
            model = load_new_model()
            df, scaler = load_dataset_and_scaler()
            
            if model is None or df is None or scaler is None:
                return render_template('result.html', stress_level="Error", advice="Terjadi kesalahan saat memuat data atau model.", emoticon="😞", error_message="Gagal memuat model atau dataset. Pastikan path sudah benar.")

            predicted_class = predict_stress(model, scaler, user_input)

            if predicted_class == 0:
                stress_level = "Ringan"
                emoticon = "😊"
                advice = ("Kamu cuma perlu istirahat sebentar. Coba santai sejenak, dengar musik favorit, atau jalan-jalan kecil. Kalau ada waktu, kamu juga bisa lakukan hobi yang bikin kamu happy, seperti baca buku, nonton film lucu atau berkumpul sama teman. Jangan lupa minum cukup air dan makan makanan yang sehat ya. Hal kecil ini bisa bikin kamu merasa lebih segar dan siap menghadapi aktivitas lagi.")
            elif predicted_class == 1:
                stress_level = "Sedang"
                emoticon = "😐"
                advice = ("Stresnya lumayan nih. Coba deh meditasi, olahraga ringan, atau tarik napas dalam-dalam buat relaksasi. Ngobrol sama teman dekat atau keluarga juga bisa bantu kamu merasa lebih lega. Kalau ada waktu, coba keluar rumah dan nikmati udara segar, mungkin sambil jalan-jalan santai. Ingat, nggak apa-apa untuk berhenti sejenak dan fokus sama diri sendiri. Kamu nggak sendirian, dan ada banyak cara buat merasa lebih baik.")
            else:
                stress_level = "Berat"
                emoticon = "😟"
                advice = ("Kondisinya kelihatan cukup berat. Jangan ragu buat cari bantuan profesional seperti konselor, psikolog, atau terapis. Mereka bisa bantu kamu memahami apa yang kamu rasakan dan memberikan solusi yang tepat. Selain itu, coba ngobrol sama orang yang kamu percaya, seperti keluarga atau sahabat, supaya kamu nggak merasa sendirian. Lakukan hal-hal kecil yang bikin nyaman, seperti dengar musik yang menenangkan atau menulis di jurnal tentang apa yang kamu rasakan. Ingat, nggak apa-apa untuk minta bantuan. Kamu punya hak untuk merasa lebih baik.")

            return render_template('result.html', stress_level=stress_level, advice=advice, emoticon=emoticon)

        session['current_index'] = current_index
        session['user_input'] = user_input

    current_key = question_keys[current_index]
    current_value = user_input[current_index] if user_input[current_index] is not None else min_max_values[current_key][0]

    return render_template(
        'questions.html',
        question=questions[current_key],
        min_value=min_max_values[current_key][0],
        max_value=min_max_values[current_key][1],
        current_value=current_value,
        is_first=current_index == 0,
        is_last=current_index == len(question_keys) - 1,
        current_index=current_index,
        total_questions=len(question_keys)
    )

@app.route("/result")
def result():
    return render_template('result.html')

if __name__ == "__main__":
    app.run(debug=True)
