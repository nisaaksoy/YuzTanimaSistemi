# -*- coding: utf-8 -*-
"""
Created on Fri May  2 19:10:14 2025

@author: pc
"""

import os
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import Label, Button
from PIL import Image, ImageTk
import numpy as np
import logging
import absl.logging
import contextlib
from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN
import tensorflow as tf
import matplotlib.pyplot as plt

# TensorFlow retracing uyarısını engelle
tf.get_logger().setLevel('ERROR')

# --- Log susturma ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
absl.logging.set_verbosity(absl.logging.ERROR)
logging.getLogger('mediapipe').setLevel(logging.CRITICAL)

@contextlib.contextmanager
def suppress_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stderr = os.dup(2)
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(old_stderr, 2)

# --- Derin öğrenme modeli yükleniyor ---
embedder = FaceNet()
detector = MTCNN()

def extract_face(image_path, required_size=(160, 160)):
    image = Image.open(image_path)
    image = image.convert('RGB')
    pixels = np.asarray(image)
    results = detector.detect_faces(pixels)
    if not results:
        return None
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    return np.asarray(image)

def get_embedding(image_path):
    face = extract_face(image_path)
    if face is None:
        print(f"Yüz bulunamadı: {image_path}")
        return None
    return embedder.embeddings([face])[0]

def get_region_embedding(image_path, region):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(rgb_image)
    if not results:
        return None
    keypoints = results[0]['keypoints']

    if region == 'eyes':
        x1, y1 = keypoints['left_eye']
        x2, y2 = keypoints['right_eye']
    elif region == 'nose':
        x1, y1 = x2, y2 = keypoints['nose']
    elif region == 'mouth':
        x1, y1 = keypoints['mouth_left']
        x2, y2 = keypoints['mouth_right']
    else:
        return None

    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    size = 40
    x1, y1 = max(cx - size, 0), max(cy - size, 0)
    x2, y2 = cx + size, cy + size
    region_crop = rgb_image[y1:y2, x1:x2]
    if region_crop.size == 0:
        return None
    region_crop = cv2.resize(region_crop, (160, 160))
    return embedder.embeddings([region_crop])[0]

def compare_regions(img1_path, img2_path):
    genel = get_embedding(img1_path), get_embedding(img2_path)
    eyes = get_region_embedding(img1_path, 'eyes'), get_region_embedding(img2_path, 'eyes')
    nose = get_region_embedding(img1_path, 'nose'), get_region_embedding(img2_path, 'nose')
    mouth = get_region_embedding(img1_path, 'mouth'), get_region_embedding(img2_path, 'mouth')

    distances = {}
    if all(g is not None for g in genel):
        distances['Genel'] = np.linalg.norm(genel[0] - genel[1])
    if all(e is not None for e in eyes):
        distances['Göz'] = np.linalg.norm(eyes[0] - eyes[1])
    if all(n is not None for n in nose):
        distances['Burun'] = np.linalg.norm(nose[0] - nose[1])
    if all(m is not None for m in mouth):
        distances['Ağız'] = np.linalg.norm(mouth[0] - mouth[1])

    return distances

def draw_landmarks(image_path):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(rgb_image)
    if not results:
        return image

    keypoints = results[0]['keypoints']
    for name, point in keypoints.items():
        cv2.circle(image, point, 5, (0, 255, 0), -1)
        cv2.putText(image, name, (point[0]+5, point[1]-5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)
    return image

def plot_similarity_graph(similarity_scores, distances):
    labels = list(similarity_scores.keys())
    similarities = [similarity_scores[k] for k in labels]
    mesafeler = [distances[k] for k in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    bars1 = ax1.bar(x - width/2, mesafeler, width, label='Mesafe', color='gray')
    bars2 = ax1.bar(x + width/2, similarities, width, label='Benzerlik (%)', color='seagreen')

    ax1.set_ylabel('Değer')
    ax1.set_title('Bölgesel Mesafe ve Benzerlik Karşılaştırması')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()

    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

    plt.tight_layout()
    plt.show()


def show_landmarked_face(image_bgr):
    plt.figure(figsize=(6, 4))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title("İşaretlenmiş Yüz")
    plt.show()
    
    


# --- GUI sınıfı ---
class FaceCompareApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Yüz Tanıma Sistemi")
        self.root.geometry("900x650")
        self.root.configure(bg="#f0f0f5")

        self.img_path1 = ""
        self.img_path2 = ""

        self.title_label = Label(root, text="Yüz Tanıma Sistemi", font=("Verdana", 22, "bold"),
                                 bg="#f0f0f5", fg="#2c3e50")
        self.title_label.pack(pady=15)

        panel_image_path = r"C:\\Users\\pc\\Downloads\\yuz-tanima.png"
        try:
            panel_img = Image.open(panel_image_path)
            panel_img.thumbnail((700, 300))
            self.panel_photo = ImageTk.PhotoImage(panel_img)
            self.panel_label = Label(root, image=self.panel_photo, bg="#f0f0f5")
            self.panel_label.pack(pady=10)
        except Exception as e:
            print(f"Görsel yüklenemedi: {e}")

        self.label1 = Label(root, text="1. Fotoğraf Yolu: Henüz seçilmedi",
                            font=("Arial", 11), bg="#f0f0f5", fg="#333")
        self.label1.pack(pady=5)

        self.btn1 = Button(root, text="📁 1. Fotoğrafı Seç", command=self.select_image1,
                           bg="#4a90e2", fg="white", font=("Helvetica", 11, "bold"),
                           padx=10, pady=5, activebackground="#357ABD", bd=0)
        self.btn1.pack(pady=5)

        self.label2 = Label(root, text="2. Fotoğraf Yolu: Henüz seçilmedi",
                            font=("Arial", 11), bg="#f0f0f5", fg="#333")
        self.label2.pack(pady=5)

        self.btn2 = Button(root, text="🖼️ 2. Fotoğrafı Seç", command=self.select_image2,
                           bg="#e94e77", fg="white", font=("Helvetica", 11, "bold"),
                           padx=10, pady=5, activebackground="#d03a61", bd=0)
        self.btn2.pack(pady=5)

        self.compare_button = Button(root, text="🔍 Karşılaştır", command=self.run_comparison,
                                     bg="#28a745", fg="white", font=("Verdana", 12, "bold"),
                                     padx=15, pady=8, activebackground="#218838", bd=0)
        self.compare_button.pack(pady=20)

        self.result_label = Label(root, text="", font=("Arial", 13, "italic"),
                                  bg="#f0f0f5", fg="#444")
        self.result_label.pack(pady=10)

    def select_image1(self):
        path = filedialog.askopenfilename()
        if path:
            self.img_path1 = path
            self.label1.config(text=f"1. Fotoğraf: {os.path.basename(path)}", fg="#007bff")

    def select_image2(self):
        path = filedialog.askopenfilename()
        if path:
            self.img_path2 = path
            self.label2.config(text=f"2. Fotoğraf: {os.path.basename(path)}", fg="#c2185b")
            
            

    def run_comparison(self):
        if not self.img_path1 or not self.img_path2:
            messagebox.showerror("Hata", "Lütfen iki fotoğraf seçin.")
            return

        distances = compare_regions(self.img_path1, self.img_path2)

        if distances:
            print("\n--- Bölgesel Benzerlik Sonuçları ---")
            similarity_scores = {}
            for k, v in distances.items():
                similarity = np.exp(-v) * 100
                similarity_scores[k] = similarity
                print(f"{k} Mesafesi: {v:.3f} | Benzerlik: {similarity:.1f}%")

            mean_distance = np.mean(list(distances.values()))
            mean_similarity = np.exp(-mean_distance) * 100

            if mean_similarity > 50:
                result_text = "Aynı Kişi"
            else:
                result_text = "Farklı Kişi"

            print(f"\nSonuç: {result_text} - Ortalama Benzerlik: {mean_similarity:.1f}%")
            self.result_label.config(
                text=f"{result_text} - Ortalama Benzerlik: {mean_similarity:.1f}%",
                fg="#28a745" if result_text == "Aynı Kişi" else "#e74c3c"
            )

            plot_similarity_graph(similarity_scores, distances)

            img1 = draw_landmarks(self.img_path1)
            img2 = draw_landmarks(self.img_path2)

            if img1 is None or img2 is None:
                print("❗ Fotoğraflar yüklenemedi.")
                return

            height = min(img1.shape[0], img2.shape[0])
            img1_resized = cv2.resize(img1, (int(img1.shape[1] * height / img1.shape[0]), height))
            img2_resized = cv2.resize(img2, (int(img2.shape[1] * height / img2.shape[0]), height))

            combined = np.hstack((img1_resized, img2_resized))
            combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)

            plt.figure(figsize=(10, 5))
            plt.imshow(combined_rgb)
            plt.axis('off')
            plt.title("Seçilen İki Fotoğraf (İşaretli)")
            plt.tight_layout()
            plt.show()

        else:
            self.result_label.config(text="⚠️ Karşılaştırma başarısız.", fg="#e74c3c")


# --- Ana uygulama çalıştırılıyor ---
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceCompareApp(root)
    root.mainloop()