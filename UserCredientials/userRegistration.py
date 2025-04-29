import cv2
import face_recognition
import os
import pickle
import customtkinter as ctk
from tkinter import messagebox
from PIL import Image, ImageTk
import threading
from customtkinter import CTkImage  # Import CTkImage


class FaceRegistrationApp:
    def __init__(self):
        # Initialize environment
        self.initialize_environment()

        # Create main window
        self.root = ctk.CTk()
        self.root.title("Face Registration System")
        self.root.geometry("1200x700")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Variables
        self.camera_active = False
        self.capture = None
        self.current_frame = None
        self.face_detected = False
        self.tk_image = None  # To keep reference to the image

        # Create UI
        self.create_widgets()

        # Start the app
        self.root.mainloop()

    def initialize_environment(self):
        """Create all required directories and files if they don't exist"""
        self.BASE_DIR = os.path.abspath(os.path.dirname(__file__))
        self.DATA_DIR = os.path.join(self.BASE_DIR, "data")
        self.FACE_DIR = os.path.join(self.BASE_DIR, "face_data")
        self.ENCODINGS_FILE = os.path.join(self.DATA_DIR, "face_encodings.pkl")

        # Create directories if they don't exist
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.FACE_DIR, exist_ok=True)

        # Initialize face encodings file if it doesn't exist
        if not os.path.exists(self.ENCODINGS_FILE):
            with open(self.ENCODINGS_FILE, "wb") as f:
                pickle.dump({"encodings": [], "usernames": [], "user_ids": []}, f)

    def create_widgets(self):
        """Create all UI components"""
        # Main container
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Header
        self.header = ctk.CTkFrame(self.main_frame, height=80)
        self.header.pack(fill="x", pady=(0, 20))

        self.title_label = ctk.CTkLabel(
            self.header,
            text="FACE REGISTRATION SYSTEM",
            font=("Arial", 24, "bold")
        )
        self.title_label.pack(pady=20)

        # Content area
        self.content_frame = ctk.CTkFrame(self.main_frame)
        self.content_frame.pack(fill="both", expand=True)

        # Left panel - Camera preview
        self.camera_frame = ctk.CTkFrame(self.content_frame, width=600)
        self.camera_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        self.camera_label = ctk.CTkLabel(
            self.camera_frame,
            text="Camera Preview",
            font=("Arial", 16),
            height=400,
            width=600
        )
        self.camera_label.pack(pady=20)

        self.camera_status = ctk.CTkLabel(
            self.camera_frame,
            text="Camera not active",
            text_color="gray"
        )
        self.camera_status.pack()

        # Right panel - Registration form
        self.form_frame = ctk.CTkFrame(self.content_frame, width=400)
        self.form_frame.pack(side="right", fill="y", padx=10, pady=10)

        # Form title
        self.form_title = ctk.CTkLabel(
            self.form_frame,
            text="Register New User",
            font=("Arial", 18, "bold")
        )
        self.form_title.pack(pady=(20, 30))

        # User ID field
        self.user_id_label = ctk.CTkLabel(self.form_frame, text="User ID:")
        self.user_id_label.pack(anchor="w", padx=20)
        self.user_id_entry = ctk.CTkEntry(self.form_frame, placeholder_text="Enter unique ID")
        self.user_id_entry.pack(fill="x", padx=20, pady=(0, 20))

        # Username field
        self.username_label = ctk.CTkLabel(self.form_frame, text="Full Name:")
        self.username_label.pack(anchor="w", padx=20)
        self.username_entry = ctk.CTkEntry(self.form_frame, placeholder_text="Enter full name")
        self.username_entry.pack(fill="x", padx=20, pady=(0, 20))

        # Submit button
        self.submit_btn = ctk.CTkButton(
            self.form_frame,
            text="Submit Details",
            command=self.validate_details,
            height=40,
            font=("Arial", 14)
        )
        self.submit_btn.pack(fill="x", padx=20, pady=(10, 20))

        # Capture button (initially disabled)
        self.capture_btn = ctk.CTkButton(
            self.form_frame,
            text="Capture Face",
            command=self.capture_face,
            height=40,
            font=("Arial", 14),
            state="disabled",
            fg_color="gray"
        )
        self.capture_btn.pack(fill="x", padx=20, pady=(0, 20))

        # Status label
        self.status_label = ctk.CTkLabel(
            self.form_frame,
            text="Enter user details to begin registration",
            text_color="gray"
        )
        self.status_label.pack(pady=10)

        # Start camera thread
        self.start_camera()

    def start_camera(self):
        """Start the camera in a separate thread"""
        if not self.camera_active:
            self.capture = cv2.VideoCapture(0)
            self.camera_active = True
            self.camera_status.configure(text="Camera active", text_color="green")

            # Start updating camera preview
            self.update_camera()

    def update_camera(self):
        """Update the camera preview using CTkImage"""
        if self.camera_active:
            ret, frame = self.capture.read()

            if ret:
                # Convert to RGB and detect faces
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)

                # Draw rectangles around faces
                for (top, right, bottom, left) in face_locations:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                # Convert to PIL Image then to CTkImage
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img = CTkImage(light_image=img, size=(600, 400))  # Use CTkImage instead

                # Update label
                self.camera_label.configure(image=img)
                self.camera_label.image = img  # Keep reference

                # Update face detection status
                self.face_detected = len(face_locations) > 0

            # Schedule next update
            self.root.after(10, self.update_camera)

    def validate_details(self):
        """Validate user details before enabling face capture"""
        user_id = self.user_id_entry.get().strip()
        username = self.username_entry.get().strip()

        if not user_id or not username:
            messagebox.showerror("Error", "Both User ID and Full Name are required!")
            return

        # Check if user ID already exists
        try:
            with open(self.ENCODINGS_FILE, "rb") as f:
                known_faces = pickle.load(f)
                if user_id in known_faces["user_ids"]:
                    messagebox.showerror("Error", f"User ID {user_id} already exists!")
                    return
        except:
            pass

        # Enable capture button
        self.capture_btn.configure(state="normal", fg_color="#1f6aa5")
        self.status_label.configure(
            text="Position your face in frame and click 'Capture Face'",
            text_color="white"
        )

        # Disable form fields and submit button
        self.user_id_entry.configure(state="disabled")
        self.username_entry.configure(state="disabled")
        self.submit_btn.configure(state="disabled")

    def capture_face(self):
        """Capture and register the face"""
        if not self.face_detected:
            messagebox.showerror("Error", "No face detected in frame!")
            return

        # Get the current frame
        ret, frame = self.capture.read()
        if not ret:
            messagebox.showerror("Error", "Failed to capture image!")
            return

        # Get user details
        user_id = self.user_id_entry.get().strip()
        username = self.username_entry.get().strip()

        # Process the face
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        if not face_locations:
            messagebox.showerror("Error", "Face moved during capture! Try again.")
            return

        if len(face_locations) > 1:
            messagebox.showerror("Error", "Multiple faces detected! Only one face should be visible.")
            return

        # Get face encoding
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        if not face_encodings:
            messagebox.showerror("Error", "Failed to encode face! Try again.")
            return

        # Load existing data
        try:
            with open(self.ENCODINGS_FILE, "rb") as f:
                known_faces = pickle.load(f)
        except:
            known_faces = {"encodings": [], "usernames": [], "user_ids": []}

        # Add new face data
        known_faces["encodings"].append(face_encodings[0])
        known_faces["usernames"].append(username)
        known_faces["user_ids"].append(user_id)

        # Save data
        with open(self.ENCODINGS_FILE, "wb") as f:
            pickle.dump(known_faces, f)

        # Save face image
        img_path = os.path.join(self.FACE_DIR, f"{user_id}_{username}.jpg")
        cv2.imwrite(img_path, frame)

        # Show success message
        messagebox.showinfo("Success", f"User {username} registered successfully!")

        # Reset form
        self.reset_form()

    def reset_form(self):
        """Reset the form for new registration"""
        self.user_id_entry.configure(state="normal")
        self.username_entry.configure(state="normal")
        self.user_id_entry.delete(0, "end")
        self.username_entry.delete(0, "end")
        self.submit_btn.configure(state="normal")
        self.capture_btn.configure(state="disabled", fg_color="gray")
        self.status_label.configure(
            text="Enter user details to begin registration",
            text_color="gray"
        )

    def __del__(self):
        """Clean up when closing"""
        if self.camera_active and self.capture:
            self.camera_active = False
            self.capture.release()


if __name__ == "__main__":
    app = FaceRegistrationApp()