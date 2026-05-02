import cv2
import pickle
import numpy as np
import mediapipe as mp
import time

MODEL_PATH = './checkpoint/svm_keypoints.pkl'
CONF_THRESHOLD = 0.6
STABLE_FRAMES = 30  

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

def normalize(landmarks):
    wrist = landmarks[0]
    base = landmarks[9]
    scale = ((base.x - wrist.x)**2 + (base.y - wrist.y)**2) ** 0.5
    scale = scale if scale > 1e-6 else 1.0
    row = []
    for lm in landmarks:
        row.extend([(lm.x - wrist.x) / scale, (lm.y - wrist.y) / scale, lm.z / scale])
    return row

def main():
    try:
        with open(MODEL_PATH, 'rb') as f:
            bundle = pickle.load(f)
        clf    = bundle['clf']
        scaler = bundle['scaler']
        print(f'[Model] Load Successfully Class: {clf.classes_}')
    except Exception as e:
        print(f'[Error] Can not load model weight: {e}')
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('[ERROR] Can not open camera')
        return

    window_name = 'Phase 2 — MediaPipe + SVM'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print('Warming up...')
    for _ in range(15):
        cap.read()
        time.sleep(0.01)
    print('[Demo v2] Launch Successfully! Enter q to quit')

    current_word = ''   
    word_history = []    
    last_char = None   
    stable_count = 0      
    space_done = False    

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                lms = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)

                row = normalize(lms.landmark)
                x_input = scaler.transform([row])
                probs = clf.predict_proba(x_input)[0]
                idx = probs.argmax()
                conf = probs[idx]
                char = clf.classes_[idx] if conf >= CONF_THRESHOLD else '?'

                if char == ' ':
                    if not space_done and current_word:
                        word_history.append(current_word)
                        current_word = ''
                        last_char = None
                        stable_count = 0
                    space_done = True
                elif char != '?':
                    space_done = False
                    if char == last_char:
                        stable_count += 1
                        if stable_count == STABLE_FRAMES:
                            current_word += char
                    else:
                        last_char = char
                        stable_count = 0
                else:
                    space_done = False
                    stable_count = 0

                # bounding box
                xs = [lm.x * w for lm in lms.landmark]
                ys = [lm.y * h for lm in lms.landmark]
                x1, y1 = max(0, int(min(xs))-20), max(0, int(min(ys))-20)
                x2, y2 = min(w, int(max(xs))+20), min(h, int(max(ys))+20)

                color = (0, 220, 0) if conf >= CONF_THRESHOLD else (0, 100, 255)
                if char == ' ':
                    color = (255, 200, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                display = 'SPACE' if char == ' ' else char
                label = f'{display} ({conf:.0%})'
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            else:
                stable_count = 0
                space_done = False

        except Exception as e:
            cv2.putText(frame, 'Inference Error', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
        cv2.putText(frame, f'Word : {current_word}_', (10, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 4)
        cv2.putText(frame, f'Word : {current_word}_', (10, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        history_str = '  '.join(word_history[-5:])
        cv2.putText(frame, history_str, (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3)
        cv2.putText(frame, history_str, (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 255), 1)

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    for _ in range(5): cv2.waitKey(1)
    hands.close()

if __name__ == '__main__':
    main()