import cv2
import numpy as np
import mediapipe as mp
import time
import os
from collections import deque

CAM_WIDTH, CAM_HEIGHT = 1280, 720
MAX_UNDO = 10
DRAW_THICKNESS = 10
ERASE_HOLD_SECONDS = 0.6
DEBOUNCE_SECONDS = 0.45

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0)]
color_names = ['Red', 'Green', 'Blue', 'Yellow', 'Cyan']
color_index = 0

canvas = None
undo_stack = deque(maxlen=MAX_UNDO)
prev_x, prev_y = 0, 0

last_action_time = {"color": 0, "screenshot": 0, "undo": 0}
fist_start = None

msg = ""
msg_time = 0

p_time = 0

os.makedirs("screenshots", exist_ok=True)


def fingers_up(lm):
    fingers = []
    ids = [8, 12, 16, 20]
    for tip in ids:
        fingers.append(1 if lm.landmark[tip].y < lm.landmark[tip - 2].y else 0)
    return fingers


def show_msg(text, duration=1):
    global msg, msg_time
    msg = text
    msg_time = time.time() + duration


def main():
    global canvas, color_index, prev_x, prev_y, fist_start, p_time

    cap = cv2.VideoCapture(0)
    cap.set(3, CAM_WIDTH)
    cap.set(4, CAM_HEIGHT)

    canvas = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)

    with mp_hands.Hands(max_num_hands=1,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.7) as hands:

        while True:
            success, img = cap.read()
            if not success:
                continue

            img = cv2.flip(img, 1)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            now = time.time()

            if result.multi_hand_landmarks:
                lm = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(img, lm, mp_hands.HAND_CONNECTIONS)

                fingers = fingers_up(lm)
                x = int(lm.landmark[8].x * CAM_WIDTH)
                y = int(lm.landmark[8].y * CAM_HEIGHT)

                # FIST → ERASE
                if sum(fingers) == 0:
                    if fist_start is None:
                        fist_start = now
                    elif now - fist_start >= ERASE_HOLD_SECONDS:
                        undo_stack.append(canvas.copy())
                        canvas[:] = 0
                        show_msg("Canvas Cleared", 1)
                        fist_start = None
                else:
                    fist_start = None

                    # TWO FINGERS → CHANGE COLOR
                    if fingers == [1, 1, 0, 0] and (now - last_action_time["color"]) > DEBOUNCE_SECONDS:
                        color_index = (color_index + 1) % len(colors)
                        show_msg(f"Color: {color_names[color_index]}", 1)
                        last_action_time["color"] = now

                    # ONE FINGER → DRAW
                    elif fingers == [1, 0, 0, 0]:
                        if prev_x == 0 and prev_y == 0:
                            undo_stack.append(canvas.copy())
                        cv2.line(canvas, (prev_x, prev_y), (x, y), colors[color_index], DRAW_THICKNESS)
                        prev_x, prev_y = x, y
                        cv2.circle(img, (x, y), 10, colors[color_index], -1)

                    # ALL FOUR FINGERS → SCREENSHOT
                    elif sum(fingers) == 4 and (now - last_action_time["screenshot"]) > DEBOUNCE_SECONDS:
                        combined = cv2.addWeighted(img, 0.5, canvas, 0.5, 0)
                        filename = f"screenshots/whiteboard_{int(time.time())}.png"
                        cv2.imwrite(filename, combined)
                        show_msg("Screenshot Saved", 1.2)
                        last_action_time["screenshot"] = now

                    # THREE FINGERS → UNDO
                    elif fingers == [1, 1, 1, 0] and (now - last_action_time["undo"]) > DEBOUNCE_SECONDS:
                        if undo_stack:
                            canvas[:] = undo_stack.pop()
                            show_msg("Undo", 0.8)
                        else:
                            show_msg("Nothing to Undo", 0.8)
                        last_action_time["undo"] = now

                    else:
                        prev_x, prev_y = 0, 0

            combined = cv2.addWeighted(img, 0.5, canvas, 0.5, 0)

            if msg and time.time() < msg_time:
                cv2.putText(combined, msg, (40, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

            c_time = time.time()
            fps = 1 / (c_time - p_time) if c_time != p_time else 0
            p_time = c_time

            cv2.putText(combined, f"FPS: {int(fps)}",
                        (CAM_WIDTH - 150, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.putText(combined, f"Color: {color_names[color_index]}",
                        (10, CAM_HEIGHT - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, colors[color_index], 3)

            cv2.imshow("Virtual Whiteboard (Python 3.11)", combined)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
