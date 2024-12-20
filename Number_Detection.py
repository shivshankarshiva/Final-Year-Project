import cv2
import mediapipe as mp
import random
import time

# Initialize hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Start capturing video from the default camera
cap = cv2.VideoCapture(1)

# Initialize game variables
current_number = None
player_number = None
last_number_change_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe hands model
    results = hands.process(rgb_frame)

    # Count the number of fingers using hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            finger_count = 0
            for finger_tip in [3, 8, 11, 15, 20]:
                finger_tip_landmark = hand_landmarks.landmark[finger_tip]
                if finger_tip_landmark.y < hand_landmarks.landmark[finger_tip - 2].y:
                    finger_count += 1

            # Set the player number based on the finger count
            player_number = finger_count

            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Change the number every 5 seconds
    if time.time() - last_number_change_time > 5:
        current_number = random.randint(0, 5)
        last_number_change_time = time.time()

    # Display current number on the screen
    if current_number is not None:
        cv2.putText(frame, f"Number: {current_number}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display player's number on the screen
    if player_number is not None:
        cv2.putText(frame, f"Your Number: {player_number}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Number Recognition Game', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
