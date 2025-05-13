# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
import cv2
from sklearn.metrics.pairwise import cosine_similarity
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")  # Uyarıları gizle

embedder = FaceNet()
detector = MTCNN()
face_data = []
tk_images = []

def extract_face_embeddings(image_path):
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)
    embeddings = []
    faces = []

    for res in results:
        x, y, w, h = res['box']
        x, y = max(0, x), max(0, y)
        face = img_rgb[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (160, 160))
        embedding = embedder.embeddings([face_resized])[0]
        embeddings.append(embedding)
        faces.append((face, res['box']))

    return embeddings, faces, img_rgb

def calculate_similarity(embedding1, embedding2):
    cos_sim = cosine_similarity([embedding1], [embedding2])[0][0]
    euclidean = np.linalg.norm(embedding1 - embedding2)
    return cos_sim, euclidean

def compare_faces():
    if len(face_data) < 2:
        messagebox.showerror("Hata", "En az iki yüz resmi seçmelisiniz!")
        return

    try:
        threshold = float(entry_threshold.get())
    except ValueError:
        messagebox.showerror("Hata", "Geçerli bir eşik değeri girin.")
        return

    text_result.delete(1.0, tk.END)
    text_result.insert(tk.END, "Karşılaştırma Sonuçları:\n\n")

    labels = []
    sims = []

    for i in range(len(face_data)):
        for j in range(i+1, len(face_data)):
            name1, emb1 = face_data[i]
            name2, emb2 = face_data[j]
            sim, dist = calculate_similarity(emb1, emb2)
            label = "AYNI KİŞİ" if sim >= threshold else "FARKLI"

            labels.append(f"{i+1}-{j+1}")
            sims.append(sim * 100)

            text_result.insert(tk.END,
                f"{name1} <-> {name2}\n"
                f"  Benzerlik: %{sim*100:.2f} | Mesafe: {dist:.4f} | Sonuç: {label}\n\n"
            )

    if labels:
        plt.figure(figsize=(8, 5))
        bars = plt.bar(labels, sims, color='skyblue')
        plt.axhline(y=threshold * 100, color='red', linestyle='--', label='Eşik Değeri')
        plt.title("Yüz Benzerlik Oranları")
        plt.xlabel("Yüz Karşılaştırmaları")
        plt.ylabel("Benzerlik (%)")
        plt.ylim(0, 100)
        plt.legend()
        plt.tight_layout()
        plt.show()

def select_multiple_photos():
    clear_all()  # önceki verileri sil

    file_paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not file_paths:
        return

    for path in file_paths:
        embeddings, faces_boxes, original_img = extract_face_embeddings(path)
        if not embeddings:
            print(f"{os.path.basename(path)} - Yüz bulunamadı.")
            continue

        if len(embeddings) > 1:
            print(f"\n📷 {os.path.basename(path)} içinde {len(embeddings)} yüz bulundu. Karşılaştırılıyor:")
            for i in range(len(embeddings)):
                for j in range(i+1, len(embeddings)):
                    sim, dist = calculate_similarity(embeddings[i], embeddings[j])
                    print(f"Yüz {i+1} <-> Yüz {j+1} | Benzerlik: %{sim*100:.2f} | Mesafe: {dist:.4f}")

        for idx, emb in enumerate(embeddings):
            face_data.append((f"{os.path.basename(path)} - Yüz {idx+1}", emb))

        show_faces_with_boxes(original_img, faces_boxes, os.path.basename(path))
        display_image_in_panel(path)


def show_faces_with_boxes(img, faces_with_boxes, title):
    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.imshow(img)
    ax.set_title(title)
    ax.axis('off')

    for face, box in faces_with_boxes:
        x, y, w, h = box
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

    plt.tight_layout()
    plt.show()

def display_image_in_panel(path):
    img = Image.open(path)
    img.thumbnail((100, 100))
    img = ImageTk.PhotoImage(img)
    tk_images.append(img)

    lbl = tk.Label(frame_images, image=img)
    lbl.image = img
    lbl.pack(side=tk.LEFT, padx=5)
    
def clear_all():
    # Görsel panelini temizle
    for widget in frame_images.winfo_children():
        widget.destroy()
    tk_images.clear()

    # Önceki yüz verilerini temizle
    face_data.clear()

    # Sonuç metnini temizle
    text_result.delete(1.0, tk.END)


# === Arayüz Kurulumu ===
root = tk.Tk()
root.title("Yüz Karşılaştırma Paneli")
root.geometry("950x700")
root.resizable(True, True)

label_title = tk.Label(root, text="Yüz Tanıma Sistemi", font=("Arial", 20, "bold"), bg='white', fg='darkblue')
label_title.pack(pady=10)

# Tanıtım görseli
image_path = r"C:\\Users\\pc\\Downloads\\yuz-tanima.png"
if os.path.exists(image_path):
    image = Image.open(image_path).resize((500, 300))
    photo = ImageTk.PhotoImage(image)
    label_image = tk.Label(root, image=photo, bg='white')
    label_image.image = photo
    label_image.pack(pady=8)

# Butonlar
frame_top = tk.Frame(root)
frame_top.pack(pady=8)

btn_add = tk.Button(frame_top, text="Çoklu Fotoğraf Seç", command=select_multiple_photos)
btn_add.pack(side=tk.LEFT, padx=8)

btn_compare = tk.Button(frame_top, text="Karşılaştır", command=compare_faces)
btn_compare.pack(side=tk.LEFT, padx=8)

# Eşik değeri giriş
label_thresh = tk.Label(frame_top, text="Benzerlik Eşiği (0-1):")
label_thresh.pack(side=tk.LEFT, padx=(20, 5))
entry_threshold = tk.Entry(frame_top, width=5)
entry_threshold.insert(0, "0.5")
entry_threshold.pack(side=tk.LEFT)

# Görsel panel
frame_images = tk.Frame(root)
frame_images.pack(pady=10)

# Sonuç metni
text_result = tk.Text(root, height=15, width=100)
text_result.pack(pady=10)

root.mainloop()
