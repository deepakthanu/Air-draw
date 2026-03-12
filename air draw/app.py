import cv2
import mediapipe as mp
import numpy as np

# Mediapipe hand setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

# Canvas
canvas = None

# Previous finger position
prev_x, prev_y = 0, 0

def fingers_up(hand_landmarks):
    fingers = []

    # Thumb
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]

    for tip, pip in zip(tips, pips):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers


while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            h, w, c = frame.shape

            x1 = int(handLms.landmark[8].x * w)
            y1 = int(handLms.landmark[8].y * h)

            fingers = fingers_up(handLms)

            # Only index finger up → draw
            if fingers[1] == 1 and sum(fingers) == 1:

                cv2.circle(frame, (x1, y1), 8, (0, 255, 0), -1)

                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x1, y1

                cv2.line(canvas, (prev_x, prev_y), (x1, y1), (255, 0, 255), 5)

                prev_x, prev_y = x1, y1

            # Thumb up → erase
            elif fingers[0] == 1 and sum(fingers) == 1:

                cv2.circle(frame, (x1, y1), 15, (0, 0, 255), -1)

                cv2.circle(canvas, (x1, y1), 40, (0, 0, 0), -1)

                prev_x, prev_y = 0, 0

            else:
                prev_x, prev_y = 0, 0

    # Merge canvas and frame
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)

    inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)

    frame = cv2.bitwise_and(frame, inv)
    frame = cv2.bitwise_or(frame, canvas)

    cv2.imshow("Air Drawing", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()