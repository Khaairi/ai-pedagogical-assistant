import streamlit as st
import pandas as pd
import cv2
import tempfile
import matplotlib.pyplot as plt
import time
import requests
from langchain_google_genai import ChatGoogleGenerativeAI

# API URL 
API_URL = "http://127.0.0.1:8000/predict"

# ====== MODUL INPUT ======
def input_data():
    uploaded_video = st.file_uploader("Unggah Video Pembelajaran", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        return tfile.name
    return None

# ====== MODUL GRAFIK ======
def display_plot(placeholder, timestamps, expressions, emotion_list):
    if not timestamps:
        return
    df = pd.DataFrame({'timestamp': timestamps, 'expression': expressions})
    df['detik'] = df['timestamp'].round().astype(int)
    summary_df = df.groupby(['detik', 'expression']).size().unstack(fill_value=0)
    summary_df = summary_df.reindex(columns=emotion_list, fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    for emotion in summary_df.columns:
        ax.plot(summary_df.index, summary_df[emotion], marker='o', linestyle='-', label=emotion)
    ax.set_title("Frekuensi Ekspresi Wajah")
    ax.legend()
    ax.grid(True)
    placeholder.pyplot(fig)
    plt.close(fig)

# ====== MODUL HASIL DAN UNDUH ======
def display_and_download_results(timestamps, expressions):
    config = {
        "emotion_dict": {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}
    }
    emotion_list = [config["emotion_dict"][i] for i in range(len(config["emotion_dict"]))]
    st.header("Hasil Analisis")

    if not timestamps or not expressions:
        st.warning("Tidak ada ekspresi yang terdeteksi untuk ditampilkan.")
        return

    df = pd.DataFrame({'timestamp': timestamps, 'expression': expressions})
    df['detik'] = df['timestamp'].round().astype(int)
    summary_df = df.groupby(['detik', 'expression']).size().unstack(fill_value=0)
    summary_df = summary_df.reindex(columns=emotion_list, fill_value=0)

    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=True).encode('utf-8')

    csv_data = convert_df_to_csv(summary_df)
    st.download_button(
       label="Unduh Hasil sebagai CSV",
       data=csv_data,
       file_name='hasil_analisis_ekspresi.csv',
       mime='text/csv',
    )

    return summary_df

# ====== MODUL PEMROSESAN VIA API ======
def process_video_via_api(video_path, image_placeholder, plot_placeholder):
    cap = cv2.VideoCapture(video_path)
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps_video)
    frame_idx = 0
    timestamps, expressions = [], []
    
    emotion_list = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            start_time = time.time()
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
            
            try:
                files = {"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")}
                response = requests.post(API_URL, files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    
                    for res in results:
                        box = res["box"]
                        label = res["label"]
                        score = res["score"]
                        
                        timestamps.append(frame_idx / fps_video)
                        expressions.append(label)

                        # Draw Box
                        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label}: {score:.2f}", (box[0], box[1]-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
            except Exception as e:
                st.error(f"API Error: {e}")

            # Update UI
            processing_fps = 1.0 / (time.time() - start_time)
            cv2.putText(frame, f"FPS: {processing_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            image_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            display_plot(plot_placeholder, timestamps, expressions, emotion_list)
        
        frame_idx += 1

    cap.release()
    return timestamps, expressions

# ====== MAIN APP ======
def main():
    st.set_page_config(layout="wide")
    st.title("Demo Pengenalan Ekspresi (Architecture: Client-Server)")

    if 'hasil_analisis' not in st.session_state:
        st.session_state['hasil_analisis'] = None

    video_path = input_data()
    
    if video_path:
        st.video(video_path)
        if st.button("Proses Video"):
            col1, col2 = st.columns([5, 3])
            image_ph = col1.empty()
            plot_ph = col2.empty()
            
            with st.spinner("Menghubungkan ke API Inference..."):
                timestamps, expressions = process_video_via_api(video_path, image_ph, plot_ph)
            
            # ... (Logika Download dan LangChain tetap sama di sini) ...
            st.success("Pemrosesan selesai!")
            summary_df = display_and_download_results(timestamps, expressions)
            st.session_state['hasil_analisis'] = summary_df
    if st.session_state['hasil_analisis'] is not None:
        df = st.session_state['hasil_analisis']
        st.write("Pratinjau Data Hasil Analisis:")
        st.dataframe(df)

        st.sidebar.header("Tanya Data")
        api_key = st.sidebar.text_input("Masukkan Google API Key", type="password")
        if api_key:
            if st.button("Generate Laporan Analisis & Solusi"):
                with st.spinner("Sedang menganalisis pola emosi siswa..."):
                    try:
                        # Inisialisasi LLM
                        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

                        data_string = df.to_csv(index=True)
                        
                        prompt_template = f"""
                            Bertindaklah sebagai Konsultan Psikologi Pendidikan & Data Analyst Senior.
                            Berikut adalah data hasil deteksi ekspresi wajah siswa per detik dalam sebuah kelas.

                            DATA (Format CSV):
                            ```csv
                            {data_string}
                            ```
                            
                            TUGAS:
                            Analisis data di atas dan buat laporan pedagogis. Jangan menghitung manual secara presisi matematika, tapi lihatlah tren dan pola datanya.
                            Lakukan langkah berikut:
                            1. **Ringkasan Keterlibatan**: Hitung persentase rata-rata ekspresi positif (Happy, Neutral, Surprise) vs negatif (Angry, Disgust, Fear, Sad).
                            2. **Identifikasi Masalah**: Cari di menit/detik keberapa terjadi lonjakan ekspresi negatif (Bosan/Sad atau Bingung/Fear) tertinggi.
                            3. **Interpretasi**: Jelaskan kemungkinan penyebabnya (misal: materi terlalu sulit, metode monoton, dll).
                            4. **Rekomendasi Solusi**: Berikan 3 saran konkret untuk guru berdasarkan data tersebut (misal: Ice breaking, Quiz interaktif, Re-explain).
                            
                            FORMAT OUTPUT (Gunakan Bahasa Indonesia Profesional & Markdown):
                            
                            ### Ringkasan Kondisi Kelas
                            [Tulis ringkasan data di sini]
                            
                            ### Titik Perhatian
                            [Sebutkan detik/waktu spesifik masalah muncul]
                            
                            ### Interpretasi & Rekomendasi
                            [Tulis analisis dan solusi poin per poin]
                            """
                        
                        response = llm.invoke(prompt_template)

                        st.markdown(response.content)
                        
                        st.session_state['last_ai_report'] = response.content
                    except Exception as e:
                            st.error(f"Gagal melakukan analisis AI: {e}")
            elif 'last_ai_report' in st.session_state:
                st.markdown(st.session_state['last_ai_report'])
        else:
            st.warning("Silakan masukkan Google API Key di sidebar untuk mengaktifkan fitur AI.")

if __name__ == "__main__":
    main()