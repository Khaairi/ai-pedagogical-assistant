import streamlit as st
import pandas as pd
import cv2
import torch
import tempfile
import matplotlib.pyplot as plt
from retinaface import RetinaFace
from torchvision import transforms
from torch import nn
from PIL import Image
import time
from model import ViTMobilenet
from huggingface_hub import hf_hub_download
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# ====== MODUL INSIALISASI ======
@st.cache_resource
def initialize():
    config = {
        "DEVICE": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        "IMG_SIZE": 224,
        "NUM_CLASSES": 7,
        "emotion_dict": {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}
    }
    config["emotion_list"] = [config["emotion_dict"][i] for i in range(len(config["emotion_dict"]))]
    config["transform"] = transforms.Compose([
        transforms.Resize((config["IMG_SIZE"], config["IMG_SIZE"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    model = ViTMobilenet(
        num_classes=config["NUM_CLASSES"],
        in_channels=3,
        num_heads=12,
        embedding_dim=768,
        num_transformer_layers=12,
        mlp_size=3072
    )

    repo_id = "MoKhaa/Hybrid_MobileNetV3_ViT"
    filename = "hybrid_mobilenet_vit_pooling_SAM_best.pt"
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)

    checkpoint = torch.load(model_path, map_location=config["DEVICE"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().to(config["DEVICE"])

    return config, model

# ====== MODUL INPUT DAN VALIDASI ======
def input_data():
    uploaded_video = st.file_uploader("Unggah Video Pembelajaran", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        video_path = tfile.name
        return video_path

# ====== MODUL GRAFIK ======
def display_plot(placeholder, timestamps, expressions, emotion_list):
    if not timestamps:
        return

    # Hitung jumlah setiap emosi per detik
    df = pd.DataFrame({'timestamp': timestamps, 'expression': expressions})
    df['detik'] = df['timestamp'].round().astype(int)
    
    # Buat pivot table: baris adalah detik, kolom adalah emosi, nilai adalah hitungan
    summary_df = df.groupby(['detik', 'expression']).size().unstack(fill_value=0)
    
    # Pastikan semua kolom emosi ada, walaupun nilainya 0
    summary_df = summary_df.reindex(columns=emotion_list, fill_value=0)

    # Membuat Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot setiap emosi sebagai garis terpisah
    for emotion in summary_df.columns:
        ax.plot(summary_df.index, summary_df[emotion], marker='o', linestyle='-', label=emotion)

    # Menyesuaikan label dan tampilan grafik
    ax.set_title("Frekuensi Ekspresi Wajah Berdasarkan Waktu", fontsize=14)
    ax.set_xlabel("Detik", fontsize=12)
    ax.set_ylabel("Jumlah Deteksi", fontsize=12) # Label Y diubah menjadi frekuensi
    ax.legend(title="Emosi")
    ax.grid(True, linestyle='--', linewidth=0.5)
    
    # Atur sumbu x agar menampilkan angka integer jika memungkinkan
    if len(summary_df.index) > 0:
        ax.set_xticks(range(summary_df.index.min(), summary_df.index.max() + 1))

    plt.tight_layout()
    
    # Tampilkan plot di placeholder Streamlit
    placeholder.pyplot(fig)
    plt.close(fig)

# ====== MODUL PEMROSESAN ======
def process_video(video_path, model, config, image_placeholder, plot_placeholder):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Gagal membuka file video.")
        return [], []

    fps_video = cap.get(cv2.CAP_PROP_FPS)
    # menentukan FPS video jadi hanya diproses 1 frame 1 detik
    frame_interval = int(fps_video)
    frame_idx = 0
    timestamps, expressions = [], []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            start_time = time.time()
            # ubah ke RGB (PIL)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            try:
                # deteksi wajah menggunakan RetinaFace
                faces = RetinaFace.detect_faces(rgb_frame)
            except Exception as e:
                # Lewati frame jika RetinaFace gagal
                frame_idx += 1
                continue

            face_tensors, boxes = [], []
            if isinstance(faces, dict):
                for face_key, face_data in faces.items():
                    # crop wajah
                    x1, y1, x2, y2 = face_data["facial_area"]
                    face_crop = rgb_frame[y1:y2, x1:x2]
                    try:
                        # transformasi agar sesuai untuk model klasifikasi ekspresi
                        tensor = config["transform"](Image.fromarray(face_crop))
                        face_tensors.append(tensor)
                        boxes.append([x1, y1, x2, y2])
                    except:
                        continue # Lewati jika crop wajah bermasalah

            if face_tensors:
                # gabungkan semua wajah jadi satu batch
                batch_tensor = torch.stack(face_tensors).to(config["DEVICE"])
                with torch.no_grad():
                    # klasifikasi ekspresi
                    logits = model(batch_tensor)
                    # ubah ke probabilitas
                    probs = torch.softmax(logits, dim=1)
                    # kelas dengan probabilitas tertinggi
                    class_ids = torch.argmax(probs, dim=1).cpu().numpy()
                    # nilai probabilitas tertinggi
                    scores = torch.max(probs, dim=1).values.cpu().numpy()

                for i, box in enumerate(boxes):
                    label = config["emotion_dict"].get(class_ids[i], "Unknown")
                    score = scores[i]
                    timestamps.append(frame_idx / fps_video)
                    expressions.append(label)
                    
                    # gambar bbox dan kelas ekspresi
                    label_text = f"{label}: {score:.2f}"
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                    cv2.putText(frame, label_text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Update tampilan
            processing_fps = 1.0 / (time.time() - start_time)
            cv2.putText(frame, f"FPS: {processing_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            image_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            display_plot(plot_placeholder, timestamps, expressions, config["emotion_list"])
        
        frame_idx += 1

    cap.release()
    return timestamps, expressions

# ====== MODUL HASIL DAN UNDUH ======
def display_and_download_results(timestamps, expressions, emotion_list):
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

# ====== MODUL UI DAN FUNGSI UTAMA ======
def main():
    st.set_page_config(layout="wide")
    st.title("Demo Pengenalan Ekspresi Wajah dari Video Pembelajaran")

    if 'hasil_analisis' not in st.session_state:
        st.session_state['hasil_analisis'] = None

    # Inisialisasi
    config, model = initialize()
    print(config['DEVICE'])
    
    # Input data
    video_path = input_data()

    # Menampilkan video yang diunggah
    st.video(video_path)

    # proses
    if st.button("Proses Video"):
        col1, col2 = st.columns([5, 3])
        image_placeholder = col1.empty()
        plot_placeholder = col2.empty()
        
        with st.spinner("Memproses video... Mohon tunggu."):
            timestamps, expressions = process_video(
                video_path, model, config, image_placeholder, plot_placeholder
            )
        if not expressions:
            return
        
        st.success("Pemrosesan selesai!")
        
        # Menampilkan Hasil Akhir
        summary_df = display_and_download_results(timestamps, expressions, config["emotion_list"])
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
