import pickle
import tkinter as tk
from tkinter import messagebox
from hashlib import sha256
import json
import cv2
import mediapipe as mp
import numpy as np
import pygame
import sys

# Initialize Pygame mixer
pygame.mixer.init()

# Dummy user database (will be stored in a JSON file)
USERS_FILE = "users.json"
users = {}

def load_users():
    try:
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)

def register():
    username = entry_username.get()
    password = entry_password.get()
    confirm_password = entry_confirm_password.get()

    if username in users:
        messagebox.showerror("Registration Failed", "Username already exists.")
        return

    if password != confirm_password:
        messagebox.showerror("Registration Failed", "Passwords do not match.")
        return

    users[username] = sha256(password.encode()).hexdigest()
    save_users(users)
    messagebox.showinfo("Registration Successful", "Registration successful. You can now login.")
    switch_to_login()

def login():
    global logged_in
    username = entry_username.get()
    password = entry_password.get()

    if username not in users or users[username] != sha256(password.encode()).hexdigest():
        messagebox.showerror("Login Failed", "Invalid username or password.")
        return

    messagebox.showinfo("Login Successful", "Welcome, " + username + "!")
    logged_in = True
    open_camera_window()

def open_camera_window():
    # Hide the login window
    root.withdraw()

    cap = cv2.VideoCapture(0)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    labels_dict = {0: 'Bathroom', 1: 'No', 2: 'Hello', 3:'Sorry', 4:'Stop', 5:'What', 6:'Help', 7:'Where', 8:'Correct', 9:'Cow', 10:'Telephone', 11:'Thank You', 12:'I Love You', 13:'Eat', 14:'Yes', 15:'Fever'}

    # Load audio files
    audio_files = {
        'Bathroom': pygame.mixer.Sound('./audio/bathroom.mp3'),
        'No': pygame.mixer.Sound('./audio/no.mp3'),
        'Hello': pygame.mixer.Sound('./audio/hello.mp3'),
        'Sorry': pygame.mixer.Sound('./audio/sorry.mp3'),
        'Stop': pygame.mixer.Sound('./audio/stop.mp3'),
        'What': pygame.mixer.Sound('./audio/what.mp3'),
        'Help': pygame.mixer.Sound('./audio/help.mp3'),
        'Where': pygame.mixer.Sound('./audio/where.mp3'),
        'Correct': pygame.mixer.Sound('./audio/correct.mp3'),
        'Cow': pygame.mixer.Sound('./audio/cow.mp3'),
        'Telephone': pygame.mixer.Sound('./audio/telephone.mp3'),
        'Thank You': pygame.mixer.Sound('./audio/thankyou.mp3'),
        'I Love You': pygame.mixer.Sound('./audio/iloveyou.mp3'),
        'Eat': pygame.mixer.Sound('./audio/eat.mp3'),
        'Yes': pygame.mixer.Sound('./audio/yes.mp3'),
        'Fever': pygame.mixer.Sound('./audio/fever.mp3'),
    }

    # Variable to keep track of whether a sound is currently playing
    sound_playing = False
    last_predicted_character = None

    while True:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            if len(data_aux) != 42:
                warning_message = "Warning: Detected " + str(len(data_aux)) + " features. Expected 42 features."
                cv2.putText(frame, warning_message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            else:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]
                if predicted_character in audio_files and not sound_playing and predicted_character != last_predicted_character:
                    audio_files[predicted_character].play()
                    sound_playing = True
                    last_predicted_character = predicted_character

                if sound_playing and not pygame.mixer.get_busy():
                    sound_playing = False
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)

        cv2.imshow('Camera Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)  # Quit the application

def switch_to_register():
    root.title("Register")
    btn_submit.config(command=register)
    btn_submit.config(text="Register")
    label_confirm_password.grid(row=2, column=0, sticky="e", pady=5)
    entry_confirm_password.grid(row=2, column=1, padx=5, pady=5)
    btn_switch.config(text="Click to Login", command=switch_to_login)

def switch_to_login():
    root.title("Login")
    btn_submit.config(command=login)
    btn_submit.config(text="Login")
    label_confirm_password.grid_forget()  # Remove the Confirm Password label and entry
    entry_confirm_password.grid_forget()
    btn_switch.config(text="New user, Click to Register", command=switch_to_register)

# Load existing users
users = load_users()

# Load model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Create the main window
root = tk.Tk()
root.title("Login")

# Create a frame for the login window
login_frame = tk.Frame(root)
login_frame.pack(padx=20, pady=20)

# Username label and entry
label_username = tk.Label(login_frame, text="Username:")
label_username.grid(row=0, column=0, sticky="e", pady=5)
entry_username = tk.Entry(login_frame)
entry_username.grid(row=0, column=1, padx=5, pady=5)

# Password label and entry
label_password = tk.Label(login_frame, text="Password:")
label_password.grid(row=1, column=0, sticky="e", pady=5)
entry_password = tk.Entry(login_frame, show="*")
entry_password.grid(row=1, column=1, padx=5, pady=5)

# Confirm Password label and entry (initially hidden)
label_confirm_password = tk.Label(login_frame, text="Confirm Password:")
entry_confirm_password = tk.Entry(login_frame, show="*")

# Submit button
btn_submit = tk.Button(login_frame, text="Login", command=login)
btn_submit.grid(row=3, columnspan=2, pady=10)

# Switch button
btn_switch = tk.Button(login_frame, text="New user, Click to Register", command=switch_to_register)
btn_switch.grid(row=4, columnspan=2, pady=5)

# Start with login page
switch_to_login()

root.mainloop()
