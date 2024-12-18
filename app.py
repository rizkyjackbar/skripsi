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

# Fungsi untuk mendapatkan pertanyaan sesuai dengan faktor
def get_question_for_factor(factor, min_val, max_val):
    questions = {
        'anxiety_level': f"Seberapa besar tingkat kecemasan yang Anda rasakan saat ini? (Rentang: {min_val} - {max_val})",
        'self_esteem': f"Seberapa besar rasa percaya diri yang Anda miliki saat ini? (Rentang: {min_val} - {max_val})",
        'mental_health_history': f"Apakah Anda memiliki riwayat masalah kesehatan mental? (Rentang: {min_val} - {max_val}) (0 = Tidak, 1 = Ya)",
        'depression': f"Seberapa besar tingkat depresi yang Anda rasakan saat ini? (Rentang: {min_val} - {max_val})",
        'headache': f"Seberapa sering Anda merasa sakit kepala? (Rentang: {min_val} - {max_val})",
        'blood_pressure': f"Seberapa tinggi tekanan darah Anda saat ini? (Rentang: {min_val} - {max_val})",
        'sleep_quality': f"Seberapa baik kualitas tidur Anda? (Rentang: {min_val} - {max_val})",
        'breathing_problem': f"Apakah Anda mengalami masalah pernapasan? (Rentang: {min_val} - {max_val})",
        'noise_level': f"Seberapa berisik lingkungan sekitar Anda? (Rentang: {min_val} - {max_val})",
        'living_conditions': f"Seberapa baik kondisi tempat tinggal Anda? (Rentang: {min_val} - {max_val})",
        'safety': f"Seberapa aman lingkungan tempat tinggal Anda? (Rentang: {min_val} - {max_val})",
        'basic_needs': f"Seberapa tercukupi kebutuhan dasar Anda saat ini? (Rentang: {min_val} - {max_val})",
        'academic_performance': f"Seberapa baik kinerja akademik Anda saat ini? (Rentang: {min_val} - {max_val})",
        'study_load': f"Seberapa berat beban studi Anda saat ini? (Rentang: {min_val} - {max_val})",
        'teacher_student_relationship': f"Bagaimana hubungan Anda dengan guru atau dosen saat ini? (Rentang: {min_val} - {max_val})",
        'future_career_concerns': f"Seberapa besar kekhawatiran Anda terhadap masa depan karir Anda? (Rentang: {min_val} - {max_val})",
        'social_support': f"Seberapa besar dukungan sosial yang Anda rasakan saat ini? (Rentang: {min_val} - {max_val})",
        'peer_pressure': f"Seberapa besar tekanan dari teman sebaya yang Anda rasakan? (Rentang: {min_val} - {max_val})",
        'extracurricular_activities': f"Seberapa banyak kegiatan ekstrakurikuler yang Anda ikuti? (Rentang: {min_val} - {max_val})",
        'bullying': f"Seberapa sering Anda mengalami perundungan? (Rentang: {min_val} - {max_val})"
    }
    return questions.get(factor, "Pertanyaan tidak tersedia")

@app.route("/", methods=["GET", "POST"])
def index():
    df = pd.read_csv('StressLevelDataset.csv')
    min_max_values = get_min_max_values(df)

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
        elif predicted_class == 1:
            stress_level = "Sedang"
        else:
            stress_level = "Berat"

        # Saran berdasarkan tingkat stres dengan bahasa yang lebih santai dan digabungkan
        advices = {
            0: [
                "Kayaknya kamu cuma butuh istirahat sejenak, coba deh luangkan waktu buat kegiatan yang seru atau olahraga ringan. Coba buat rutinitas yang lebih santai biar bisa ngatur waktu lebih baik, jangan terlalu paksain diri. Masih bisa diatasi kok, coba lebih sering relaksasi, dan jaga kesehatan ya!"
            ],
            1: [
                "Mungkin kamu bisa coba meditasi atau relaksasi biar pikiran lebih tenang. Coba tidur lebih cukup, atur waktu buat belajar atau kerja, jangan sampai overthinking. Gak ada salahnya ngobrol sama teman atau konselor biar merasa lebih ringan."
            ],
            2: [
                "Mungkin udah waktunya cari dukungan profesional, kayak konsultan atau terapis, biar bisa bantu atasi masalah. Jangan ragu buat minta bantuan dari orang-orang terdekat, mereka pasti bakal siap bantu. Luangkan waktu untuk meditasi atau aktivitas lain yang bisa bikin pikiran lebih fresh, supaya gak stuck."
            ]
        }

        advice = advices[predicted_class]
        return render_template('index.html', stress_level=stress_level, advice=advice, min_max_values=min_max_values, get_question_for_factor=get_question_for_factor)

    return render_template('index.html', stress_level=None, advice=None, min_max_values=min_max_values, get_question_for_factor=get_question_for_factor)

if __name__ == "__main__":
    app.run(debug=True)
