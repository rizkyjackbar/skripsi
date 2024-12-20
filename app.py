from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pandas as pd

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Ganti dengan secret key Anda

# Fungsi untuk memuat model
def load_new_model():
    model = tf.keras.models.load_model('model/stress_level_model_v2.h5')
    return model

# Fungsi untuk memuat dataset dan scaler
def load_dataset_and_scaler():
    df = pd.read_csv('StressLevelDataset.csv')
    scaler = StandardScaler()
    X = df.drop(columns=['stress_level'])
    scaler.fit(X)
    return scaler

# Fungsi untuk memprediksi tingkat stres
def predict_stress(model, scaler, user_input):
    user_input_scaled = scaler.transform([user_input])
    prediction = model.predict(user_input_scaled)
    predicted_class = np.argmax(prediction, axis=1)
    return predicted_class[0]

# Fungsi untuk mendapatkan pertanyaan dan min-max
def get_stress_questions_and_min_max():
    df = pd.read_csv('StressLevelDataset.csv')
    questions = {
        'anxiety_level': "Seberapa cemas Anda merasa akhir-akhir ini?",
        'self_esteem': "Bagaimana Anda menilai harga diri Anda saat ini?",
        'mental_health_history': "Apakah Anda memiliki riwayat masalah kesehatan mental?",
        'depression': "Seberapa sering Anda merasa tertekan atau sedih dalam beberapa minggu terakhir?",
        'headache': "Seberapa sering Anda mengalami sakit kepala atau migrain?",
        'blood_pressure': "Bagaimana kondisi tekanan darah Anda akhir-akhir ini?",
        'sleep_quality': "Seberapa baik kualitas tidur Anda dalam beberapa minggu terakhir?",
        'breathing_problem': "Apakah Anda mengalami masalah pernapasan atau sesak napas?",
        'noise_level': "Seberapa tinggi tingkat kebisingan di lingkungan Anda (rumah, tempat kerja, sekolah)?",
        'living_conditions': "Bagaimana kondisi tempat tinggal Anda saat ini?",
        'safety': "Seberapa aman Anda merasa di lingkungan sekitar Anda?",
        'basic_needs': "Apakah kebutuhan dasar Anda (makanan, tempat tinggal, pakaian) tercukupi dengan baik?",
        'academic_performance': "Bagaimana Anda menilai performa akademis Anda?",
        'study_load': "Seberapa banyak beban studi atau pekerjaan akademis yang Anda rasakan saat ini?",
        'teacher_student_relationship': "Bagaimana hubungan Anda dengan guru atau pengajar Anda?",
        'future_career_concerns': "Seberapa khawatir Anda mengenai karier dan masa depan Anda?",
        'social_support': "Seberapa banyak dukungan yang Anda rasakan dari teman atau keluarga?",
        'peer_pressure': "Seberapa besar tekanan yang Anda rasakan dari teman sebaya atau lingkungan sosial Anda?",
        'extracurricular_activities': "Seberapa banyak kegiatan ekstrakurikuler atau kegiatan di luar sekolah/kerja yang Anda ikuti?",
        'bullying': "Apakah Anda mengalami perundungan atau bullying di sekolah atau tempat kerja?"
    }
    min_max_values = {col: (df[col].min(), df[col].max()) for col in df.columns if col != 'stress_level'}
    return questions, min_max_values

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'start' in request.form:
            questions, _ = get_stress_questions_and_min_max()
            session['current_index'] = 0
            session['user_input'] = [None] * len(questions)  # Inisialisasi dengan ukuran sesuai jumlah pertanyaan
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
        return redirect(url_for('index'))  # Jika di luar batas, kembalikan ke halaman awal

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
            scaler = load_dataset_and_scaler()
            predicted_class = predict_stress(model, scaler, user_input)

            if predicted_class == 0:
                stress_level = "Ringan"
                emoticon = "😊"
                advice = "Kayaknya kamu cuma butuh istirahat sejenak. Coba luangkan waktu buat hal yang bikin rileks."
            elif predicted_class == 1:
                stress_level = "Sedang"
                emoticon = "😐"
                advice = "Mungkin kamu bisa coba meditasi, relaksasi, dan ngobrol sama teman atau konselor."
            else:
                stress_level = "Berat"
                emoticon = "😟"
                advice = "Cari dukungan profesional seperti konsultan atau terapis. Jangan ragu minta bantuan orang terdekat."

            return render_template('result.html', stress_level=stress_level, advice=advice, emoticon=emoticon)

        session['current_index'] = current_index
        session['user_input'] = user_input

    current_key = question_keys[current_index]
    current_value = user_input[current_index] if user_input[current_index] is not None else min_max_values[current_key][0]

    # Pass `current_index` and `total_questions` to template
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

if __name__ == "__main__":
    app.run(debug=True)
