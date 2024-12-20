import cv2
import numpy as np
import os
import mediapipe as mp

# Define the options and their corresponding .py files
options = {
    1: "Fungame.py",
    2: "main.py",
    3: "Number_Detection.py"
}

def run_script(script_name):
    os.system(f"python {script_name}")

# Detect click on options using hand gestures
def detect_click(cx, cy, choice):
    for key, value in options.items():
        if cy in range(100 + 50 * key, 100 + 50 * key + 50) and cx in range(50, 300):
            choice[0] = key

# Display options to the user using OpenCV
def display_options_on_camera():
    cap = cv2.VideoCapture(1)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    choice = [0]

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                    detect_click(cx, cy, choice)

        frame = cv2.putText(frame, 'Choose an option:', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        for key, value in options.items():
            frame = cv2.putText(frame, f"{key}. {value}", (50, 100 + 50 * key), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Options', frame)

        if choice[0] != 0:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return choice[0]

# OpenCV code to capture user input
def main():
    choice = display_options_on_camera()

    if choice in options:
        selected_option = options[choice]
        run_script(selected_option)
    else:
        print("Invalid choice. Please select a valid option.")

if __name__ == '__main__':
    main()
