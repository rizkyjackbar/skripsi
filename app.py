from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pandas as pd

app = Flask(__name__)

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

# Fungsi untuk mendapatkan min-max dari dataset
def get_min_max_values(df):
    min_max_values = {col: (df[col].min(), df[col].max()) for col in df.columns if col != 'stress_level'}
    return min_max_values

# Pertanyaan terkait setiap faktor stres
def get_stress_questions():
    return {
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

@app.route("/", methods=["GET", "POST"])
def index():
    df = pd.read_csv('StressLevelDataset.csv')
    min_max_values = get_min_max_values(df)
    questions = get_stress_questions()

    stress_level = None
    advice = None
    emoticon = None

    if request.method == "POST":
        # Ambil input dari form
        user_input = []
        for col in min_max_values:
            user_input.append(float(request.form[col]))

        # Memuat model dan scaler
        model = load_new_model()
        scaler = load_dataset_and_scaler()

        # Prediksi tingkat stres
        predicted_class = predict_stress(model, scaler, user_input)

        # Tentukan tingkat stres
        if predicted_class == 0:
            stress_level = "Ringan"
            emoticon = "😊"
        elif predicted_class == 1:
            stress_level = "Sedang"
            emoticon = "😐"
        else:
            stress_level = "Berat"
            emoticon = "😟"

        # Saran berdasarkan tingkat stres
        advices = {
            0: "Kayaknya kamu cuma butuh istirahat sejenak. Coba luangkan waktu buat hal yang bikin rileks.",
            1: "Mungkin kamu bisa coba meditasi, relaksasi, dan ngobrol sama teman atau konselor.",
            2: "Cari dukungan profesional seperti konsultan atau terapis. Jangan ragu minta bantuan orang terdekat."
        }
        advice = advices[predicted_class]

    return render_template('index.html', stress_level=stress_level, advice=advice, emoticon=emoticon,
                           min_max_values=min_max_values, questions=questions)

if __name__ == "__main__":
    app.run(debug=True)
