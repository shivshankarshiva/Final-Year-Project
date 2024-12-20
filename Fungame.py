import cv2
import mediapipe as mp
import random

# Initialize hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Set up game variables
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480
object_size = 50
object_x = random.randint(0, WINDOW_WIDTH - object_size)
object_y = 0
object_speed = 5
score = 0

# Start capturing video from the default camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        continue

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB (required for hand tracking)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands using Mediapipe hand tracking
    results = hands.process(rgb_frame)

    # Draw falling object (rectangle) on the frame
    cv2.rectangle(frame, (object_x, object_y), (object_x + object_size, object_y + object_size), (0, 255, 0), -1)

    # Check hand presence and landmarks
    hand_detected = False

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            hand_landmark_points = [(int(hand_landmarks.landmark[i].x * WINDOW_WIDTH),
                                     int(hand_landmarks.landmark[i].y * WINDOW_HEIGHT)) for i in range(21)]
            hand_detected = True

            # Check if hand is catching the falling object
            if object_x < hand_landmark_points[8][0] < object_x + object_size and object_y < hand_landmark_points[8][1] < object_y + object_size:
                # Hand caught the falling object
                object_x = random.randint(0, WINDOW_WIDTH - object_size)
                object_y = 0
                score += 1

    # Display score on the frame
    cv2.putText(frame, f"Score: {score}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow('Object Detection Game', frame)

    # Move the falling object down
    object_y += object_speed

    # Reset object position if it reaches the bottom
    if object_y > WINDOW_HEIGHT:
        object_x = random.randint(0, WINDOW_WIDTH - object_size)
        object_y = 0

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close OpenCV window
cap.release()
cv2.destroyAllWindows()
