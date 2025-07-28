import cv2
import numpy as np
import mediapipe as mp
import time
import os

#  Hand Tracking Setup 
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=4, min_detection_confidence=0.7)  # Allow 4 hands
mp_draw = mp.solutions.drawing_utils

#  Drawing Config 
canvas = None
prev_x, prev_y = 0, 0
draw_color = (0, 0, 255)
brush_thickness = 10
save_count = 1

#  Virtual Button Config
buttons = {
    'Red': (20, 1),
    'Green': (140, 1),
    'Blue': (260, 1),
    'Clear': (380, 1),
    'Save': (500, 1),
    'Exit': (620, 1),
    'Thin': (740, 1),
    'Medium': (860, 1),
    'Thick': (980, 1),
}

colors = {
    'Red': (0, 0, 255),
    'Green': (0, 255, 0),
    'Blue': (255, 0, 0)
}

def draw_buttons(img):
    for label, (x, y) in buttons.items():
        color = colors[label] if label in colors else (50, 50, 50)
        cv2.rectangle(img, (x, y), (x + 100, y + 65), color, -1)
        cv2.putText(img, label, (x + 10, y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def check_button_press(x, y):
    for label, (bx, by) in buttons.items():
        if bx < x < bx + 100 and by < y < by + 65:
            return label
    return None

def fingers_up(lm_list):
    # Returns a list of 5 booleans: [thumb, index, middle, ring, pinky]
    # True if finger is up, False if down
    tips = [4, 8, 12, 16, 20]
    fingers = []
    # Thumb: compare tip and IP joint in x direction (for right hand)
    fingers.append(lm_list[tips[0]][0] > lm_list[tips[0] - 1][0])
    # Other fingers: compare tip and PIP joint in y direction
    for i in range(1, 5):
        fingers.append(lm_list[tips[i]][1] < lm_list[tips[i] - 2][1])
    return fingers

#  Webcam Setup 
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

prev_points = {}  # Store previous points for each hand

while True:
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)
    if canvas is None:
        canvas = np.zeros_like(frame)

    draw_buttons(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_idx, handLms in enumerate(result.multi_hand_landmarks):
            lm_list = []
            for id, lm in enumerate(handLms.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))

            if len(lm_list) >= 21:
                x1, y1 = lm_list[8]  # Index finger tip
                x2, y2 = lm_list[12] # Middle finger tip

                # Draw circle at index finger
                cv2.circle(frame, (x1, y1), 10, draw_color, cv2.FILLED)

                finger_states = fingers_up(lm_list)
                # Only allow drawing if index finger up and middle finger down
                if finger_states[1] and not finger_states[2]:
                    # Drawing mode
                    prev_x, prev_y = prev_points.get(hand_idx, (0, 0))
                    if prev_x == 0 and prev_y == 0:
                        prev_points[hand_idx] = (x1, y1)
                    else:
                        cv2.line(canvas, (prev_x, prev_y), (x1, y1), draw_color, brush_thickness)
                        prev_points[hand_idx] = (x1, y1)
                else:
                    prev_points[hand_idx] = (0, 0)
                    # If fingers are close = selection mode
                    if abs(y2 - y1) < 30:
                        action = check_button_press(x1, y1)
                        if action:
                            if action in colors:
                                draw_color = colors[action]
                            elif action == 'Clear':
                                canvas = np.zeros_like(frame)
                            elif action == 'Save':
                                filename = f"air_canvas{save_count}.png"
                                cv2.imwrite(filename, canvas)
                                print(f"[ Saved as {filename}]")
                                save_count += 1
                            elif action == 'Exit':
                                cap.release()
                                cv2.destroyAllWindows()
                                exit()
                            elif action == 'Thin':
                                brush_thickness = 3
                            elif action == 'Medium':
                                brush_thickness = 5
                            elif action == 'Thick':
                                brush_thickness = 10

            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    # Combine canvas and webcam
    merged = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
    cv2.putText(merged, f"Brush: {brush_thickness}px", (10, 680),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow("Air Canvas with Hand 🎨", merged)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
