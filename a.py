import os
import numpy as np
import joblib
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import cv2

# ------------------------------
# 1️⃣ **Load Saved Models**
# ------------------------------
SAVE_DIR = "/content/drive/MyDrive/saved_models"

# Load FaceNet model
facenet_model = load_model(os.path.join(SAVE_DIR, 'facenet_model.h5'))

# Load SVM classifier
svm_classifier = joblib.load(os.path.join(SAVE_DIR, 'svm_classifier.pkl'))

# Load Scaler
scaler = joblib.load(os.path.join(SAVE_DIR, 'scaler.pkl'))

# Load Label Encoder
label_encoder = joblib.load(os.path.join(SAVE_DIR, 'label_encoder.pkl'))

# ------------------------------
# 2️⃣ **Helper Function: Extract Embeddings**
# ------------------------------
def get_face_embedding(image_path):
    """Extract FaceNet embedding from the given image."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = cv2.resize(image, (160, 160))  # Resize to FaceNet input size
    image = np.expand_dims(image / 255.0, axis=0)  # Normalize & expand dimensions
    
    embedding = facenet_model.predict(image)
    embedding = scaler.transform(embedding)  # Normalize embedding
    return embedding

# ------------------------------
# 3️⃣ **GUI: Face Recognition App**
# ------------------------------
class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("500x600")
        self.root.configure(bg="#f0f0f0")

        # Title Label
        self.label_title = tk.Label(root, text="Face Recognition", font=("Arial", 18, "bold"), bg="#f0f0f0")
        self.label_title.pack(pady=10)

        # Image Label (Display Selected Image)
        self.image_label = tk.Label(root, text="No image selected", bg="white", width=50, height=15)
        self.image_label.pack(pady=10)

        # Load Image Button
        self.btn_load = tk.Button(root, text="Load Image", font=("Arial", 12), command=self.load_image)
        self.btn_load.pack(pady=5)

        # Predict Button
        self.btn_predict = tk.Button(root, text="Recognize Face", font=("Arial", 12), command=self.predict_face, state=tk.DISABLED)
        self.btn_predict.pack(pady=5)

        # Result Label
        self.result_label = tk.Label(root, text="", font=("Arial", 14, "bold"), fg="blue", bg="#f0f0f0")
        self.result_label.pack(pady=10)

    def load_image(self):
        """Open file dialog to load an image."""
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.image_path = file_path
            img = Image.open(file_path)
            img = img.resize((200, 200), Image.ANTIALIAS)
            img_tk = ImageTk.PhotoImage(img)
            self.image_label.config(image=img_tk, text="", width=200, height=200)
            self.image_label.image = img_tk
            self.btn_predict.config(state=tk.NORMAL)

    def predict_face(self):
        """Predict the face using SVM classifier."""
        try:
            embedding = get_face_embedding(self.image_path)
            prediction = svm_classifier.predict(embedding)
            person_name = label_encoder.inverse_transform(prediction)[0]
            self.result_label.config(text=f"Recognized as: {person_name}", fg="green")
        except Exception as e:
            messagebox.showerror("Error", f"Prediction Failed: {e}")

# ------------------------------
# 4️⃣ **Run the GUI**
# ------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
