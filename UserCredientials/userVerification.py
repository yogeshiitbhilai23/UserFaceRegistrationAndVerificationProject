from tkinter import messagebox

import cv2
import face_recognition
import os
import pickle
import customtkinter as ctk
from PIL import Image, ImageTk
from customtkinter import CTkImage
import numpy as np


class FaceVerificationApp:
    def __init__(self):
        # Initialize environment
        self.BASE_DIR = os.path.abspath(os.path.dirname(__file__))
        self.DATA_DIR = os.path.join(self.BASE_DIR, "data")
        self.FACE_DIR = os.path.join(self.BASE_DIR, "face_data")
        self.ENCODINGS_FILE = os.path.join(self.DATA_DIR, "face_encodings.pkl")

        # Load registered faces
        self.known_faces = self.load_known_faces()

        # Create main window
        self.root = ctk.CTk()
        self.root.title("Face Verification System")
        self.root.geometry("1000x600")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Variables
        self.camera_active = False
        self.capture = None
        self.current_user = None
        self.last_verification_time = 0
        self.verification_interval = 2  # seconds between verifications

        # Create UI
        self.create_widgets()

        # Start camera
        self.start_camera()

        # Start the app
        self.root.mainloop()

    def load_known_faces(self):
        """Load registered face encodings"""
        if not os.path.exists(self.ENCODINGS_FILE):
            messagebox.showerror("Error", "No registered faces found!")
            return {"encodings": [], "usernames": [], "user_ids": []}

        with open(self.ENCODINGS_FILE, "rb") as f:
            return pickle.load(f)

    def create_widgets(self):
        """Create all UI components"""
        # Main container
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Header
        self.header = ctk.CTkFrame(self.main_frame, height=60)
        self.header.pack(fill="x", pady=(0, 10))

        self.title_label = ctk.CTkLabel(
            self.header,
            text="FACE VERIFICATION SYSTEM",
            font=("Arial", 22, "bold")
        )
        self.title_label.pack(pady=10)

        # Content area
        self.content_frame = ctk.CTkFrame(self.main_frame)
        self.content_frame.pack(fill="both", expand=True)

        # Left panel - Camera preview
        self.camera_frame = ctk.CTkFrame(self.content_frame, width=600)
        self.camera_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        self.camera_label = ctk.CTkLabel(
            self.camera_frame,
            text="",
            font=("Arial", 16),
            height=400,
            width=600
        )
        self.camera_label.pack(pady=10)

        # Right panel - Verification info
        self.info_frame = ctk.CTkFrame(self.content_frame, width=300)
        self.info_frame.pack(side="right", fill="y", padx=10, pady=10)

        self.status_label = ctk.CTkLabel(
            self.info_frame,
            text="Status: Ready",
            font=("Arial", 16),
            text_color="white"
        )
        self.status_label.pack(pady=20)

        self.user_label = ctk.CTkLabel(
            self.info_frame,
            text="User: Not identified",
            font=("Arial", 14)
        )
        self.user_label.pack(pady=10)

        self.user_id_label = ctk.CTkLabel(
            self.info_frame,
            text="ID: -",
            font=("Arial", 14)
        )
        self.user_id_label.pack(pady=10)

        self.confidence_label = ctk.CTkLabel(
            self.info_frame,
            text="Confidence: -",
            font=("Arial", 14)
        )
        self.confidence_label.pack(pady=10)

        # Verification button
        self.verify_btn = ctk.CTkButton(
            self.info_frame,
            text="Start Verification",
            command=self.toggle_verification,
            height=40,
            font=("Arial", 14)
        )
        self.verify_btn.pack(pady=20, fill="x")

    def start_camera(self):
        """Start the camera"""
        self.capture = cv2.VideoCapture(0)
        self.camera_active = True
        self.update_camera()

    def toggle_verification(self):
        """Toggle verification state"""
        if self.verify_btn.cget("text") == "Start Verification":
            self.verify_btn.configure(text="Stop Verification", fg_color="red")
            self.status_label.configure(text="Status: Verifying...", text_color="yellow")
        else:
            self.verify_btn.configure(text="Start Verification", fg_color="#1f6aa5")
            self.status_label.configure(text="Status: Ready", text_color="white")
            self.reset_verification_display()

    def reset_verification_display(self):
        """Reset the verification display"""
        self.user_label.configure(text="User: Not identified")
        self.user_id_label.configure(text="ID: -")
        self.confidence_label.configure(text="Confidence: -")
        self.current_user = None

    def update_camera(self):
        """Update the camera preview with face verification"""
        if self.camera_active:
            ret, frame = self.capture.read()

            if ret:
                # Process frame for display
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                display_frame = frame.copy()

                # Only run verification if button is in "verifying" state
                if self.verify_btn.cget("text") == "Stop Verification":
                    # Find all face locations in the current frame
                    face_locations = face_recognition.face_locations(rgb_frame)

                    if face_locations:
                        # Get face encodings for detected faces
                        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                            # Compare with known faces
                            matches = face_recognition.compare_faces(
                                self.known_faces["encodings"],
                                face_encoding,
                                tolerance=0.5
                            )

                            name = "Unknown"
                            user_id = "-"
                            confidence = "0%"

                            if True in matches:
                                # Find the best match
                                face_distances = face_recognition.face_distance(
                                    self.known_faces["encodings"],
                                    face_encoding
                                )
                                best_match_index = np.argmin(face_distances)

                                if matches[best_match_index]:
                                    name = self.known_faces["usernames"][best_match_index]
                                    user_id = self.known_faces["user_ids"][best_match_index]
                                    confidence = f"{round((1 - face_distances[best_match_index]) * 100, 2)}%"
                                    self.current_user = name

                                    # Update UI
                                    self.user_label.configure(text=f"User: {name}")
                                    self.user_id_label.configure(text=f"ID: {user_id}")
                                    self.confidence_label.configure(text=f"Confidence: {confidence}")
                                    self.status_label.configure(text="Status: Verified", text_color="green")

                            # Draw rectangle and label
                            cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                            cv2.putText(display_frame, name, (left, top - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Convert to CTkImage for display
                img = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
                img = CTkImage(light_image=img, size=(600, 400))

                # Update label
                self.camera_label.configure(image=img)
                self.camera_label.image = img

            # Schedule next update
            self.root.after(10, self.update_camera)

    def __del__(self):
        """Clean up when closing"""
        if self.camera_active and self.capture:
            self.camera_active = False
            self.capture.release()


if __name__ == "__main__":
    app = FaceVerificationApp()
