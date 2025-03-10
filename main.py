import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Get screen size
screen_w, screen_h = pyautogui.size()
pyautogui.FAILSAFE = False  # Prevents program from stopping if the cursor hits the corner

# Open webcam
cap = cv2.VideoCapture(0)

# Variables for relative movement
prev_x, prev_y = None, None  # Store previous hand position
cursor_x, cursor_y = screen_w // 2, screen_h // 2  # Start cursor at center

last_left_click_time = 0  # Prevent excessive clicking
last_right_click_time = 0
last_screenshot_time = 0

holding_left_click = False  # Track if left click is being held

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Find the lowest point on the hand (usually wrist or bottom landmark)
            min_y = float('inf')

            lowest_point = hand_landmarks.landmark[0]

            if lowest_point:
                # Get hand position in frame
                hand_x = lowest_point.x
                hand_y = lowest_point.y

                # Convert to screen-relative movement
                if prev_x is not None and prev_y is not None:
                    dx = (hand_x - prev_x) * screen_w * 2  # Scale movement
                    dy = (hand_y - prev_y) * screen_h * 2

                    cursor_x += dx
                    cursor_y += dy

                    # Clamp values to stay within screen bounds
                    cursor_x = max(0, min(screen_w, cursor_x))
                    cursor_y = max(0, min(screen_h, cursor_y))

                    # Move cursor
                    pyautogui.moveTo(cursor_x, cursor_y, duration=0.2)

                # Update previous hand position
                prev_x, prev_y = hand_x, hand_y

            # Get key finger positions
            thumb = hand_landmarks.landmark[4]
            index_finger = hand_landmarks.landmark[8]
            middle_finger = hand_landmarks.landmark[12]
            ring_finger = hand_landmarks.landmark[16]

            # Measure pinch distances
            index_thumb_distance = np.linalg.norm([thumb.x - index_finger.x, thumb.y - index_finger.y])
            middle_thumb_distance = np.linalg.norm([thumb.x - middle_finger.x, thumb.y - middle_finger.y])
            ring_thumb_distance = np.linalg.norm([thumb.x - ring_finger.x, thumb.y - ring_finger.y])

            # Left click (pinch: index + thumb)
            if index_thumb_distance < 0.05:
                current_time = time.time()
                if current_time - last_left_click_time > 0.5:  # Prevent excessive clicking
                    pyautogui.click()
                    last_left_click_time = current_time
                    print("Left Click!")

            # Right click (pinch: middle + thumb)
            if middle_thumb_distance < 0.05:
                current_time = time.time()
                if current_time - last_right_click_time > 0.5:
                    pyautogui.rightClick()
                    last_right_click_time = current_time
                    print("Right Click!")

            # Screenshot (pinch: ring + thumb)
            if ring_thumb_distance < 0.05:
                current_time = time.time()
                if current_time - last_screenshot_time > 1:  # Prevent excessive screenshots
                    pyautogui.screenshot("screenshot.png")
                    last_screenshot_time = current_time
                    print("Screenshot taken!")

            # Hold left click (index finger bending forward)
            index_tip = np.array([index_finger.x, index_finger.y])
            index_mcp = np.array([hand_landmarks.landmark[5].x, hand_landmarks.landmark[5].y])  # Base of index finger

            index_finger_bent = index_tip[1] > index_mcp[1]  # Check if tip is lower than the base

            if index_finger_bent and not holding_left_click:
                pyautogui.mouseDown()
                holding_left_click = True
                print("Holding Left Click!")

            elif not index_finger_bent and holding_left_click:
                pyautogui.mouseUp()
                holding_left_click = False
                print("Released Left Click!")

    # Show webcam feed
    cv2.imshow("Hand Tracking Mouse", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
