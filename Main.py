import cv2
import mediapipe as mp
from pynput.keyboard import Controller, Key

# Constants
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
CENTER_X = SCREEN_WIDTH // 2
CENTER_Y = SCREEN_HEIGHT // 2
DEAD_ZONE = 38  # Dead zone for no movement
keyboard = Controller()

# Variables
last_state_right = False
last_state_left = False
last_state_up = False
last_state_down = False
last_state_above_70 = False
result_list = []
capture_count = 0  # Counter for captured images

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)

def move_right():
    keyboard.press(Key.right)
    keyboard.release(Key.right)

def move_left():
    keyboard.press(Key.left)
    keyboard.release(Key.left)

def move_up():
    keyboard.press(Key.up)
    keyboard.release(Key.up)

def move_down():
    keyboard.press(Key.down)
    keyboard.release(Key.down)

def draw_overlay(img):
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 280), (640, 480), (84, 199, 112), -1)
    cv2.rectangle(overlay, (0, 200), (640, 280), (219, 161, 61), -1)
    cv2.rectangle(overlay, (0, 0), (640, 200), (84, 199, 112), -1)
    alpha = 0.4
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    cv2.line(img, (0, 280), (640, 280), (0, 255, 0), 2)
    cv2.line(img, (0, 200), (640, 200), (0, 255, 0), 2)
    cv2.line(img, (0, 240), (280, 240), (0, 0, 0), 2)
    cv2.line(img, (360, 240), (640, 240), (0, 0, 0), 2)
    cv2.line(img, (320, 280), (320, 480), (0, 0, 0), 2)
    cv2.line(img, (320, 200), (320, 0), (0, 0, 0), 2)
    cv2.rectangle(img, (280, 280), (360, 200), (0, 0, 255), 2)
    return img

def filter_numbers(number):
    global result_list, last_state_above_70
    if number > 70:
        result_list.append(number)
        if not last_state_above_70:
            last_state_above_70 = True
            return print_max_value()
    else:
        if last_state_above_70:
            last_state_above_70 = False
            return print_max_value()
    return None

def print_max_value():
    global result_list
    if not result_list:
        print("empty")
        return None
    max_value = max(result_list)
    result_list.clear()
    return max_value

def handle_direction(x_c, y_c):
    global last_state_right, last_state_left, last_state_up, last_state_down
    x = abs(x_c)
    y = abs(y_c)

    if -DEAD_ZONE <= x_c <= DEAD_ZONE and -DEAD_ZONE <= y_c <= DEAD_ZONE:
        last_state_right = last_state_left = last_state_up = last_state_down = False
    elif x > y:
        if x_c > 0 and not last_state_left:  # Left movement
            move_left()
            last_state_left = True
            last_state_right = False
        elif x_c < 0 and not last_state_right:  # Right movement
            move_right()
            last_state_right = True
            last_state_left = False
    else:
        if y_c > 0 and not last_state_up:  # Up movement
            move_up()
            last_state_up = True
            last_state_down = False
        elif y_c < 0 and not last_state_down:  # Down movement
            move_down()
            last_state_down = True
            last_state_up = False

def capture_frame(img):
    global capture_count
    capture_filename = f"capture_{capture_count}.png"
    cv2.imwrite(capture_filename, img)
    print(f"Captured frame saved as {capture_filename}")
    capture_count += 1

def run():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    cap.set(3, SCREEN_WIDTH)
    cap.set(4, SCREEN_HEIGHT)

    while True:
        ret, img = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    if id == 8:  # Index finger tip
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(img, (cx, cy), 3, (0, 255, 0), cv2.FILLED)
                        cv2.line(img, (cx, 0), (cx, 480), (0, 0, 255), 2)
                        cv2.line(img, (0, cy), (640, cy), (0, 0, 255), 2)

                        x_c = CENTER_X - cx
                        y_c = CENTER_Y - cy
                        handle_direction(x_c, y_c)

        img = draw_overlay(img)
        cv2.imshow("Frame", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord('c'):  # Capture frame
            capture_frame(img)

    cap.release() 
    cv2.destroyAllWindows()


if __name__ == "__main__":
 
    run()
