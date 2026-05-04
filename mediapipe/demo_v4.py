import cv2
import pickle
import numpy as np
import mediapipe as mp
import time

# ══════════════════════════════════════════
#  配置
# ══════════════════════════════════════════
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

    # --- 状态控制变量 ---
    current_word = ''   
    word_history = []    
    last_char = None   
    stable_count = 0      
    action_fired = False  # 动作触发锁
    
    # 字幕控制
    subtitle_text = ""
    subtitle_until = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # 1. 翻转并获取尺寸
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # 2. 创建一个【极度模糊】的背景画面
        # 优化技巧：先将画面长宽缩小到 1/4，再进行模糊，最后放大回原尺寸。
        # 这样不仅能制造极重的“毛玻璃”隐私遮挡效果，还能大幅节省计算资源，防止画面卡顿。
        small_frame = cv2.resize(frame, (w // 4, h // 4))
        heavy_blur = cv2.GaussianBlur(small_frame, (99, 99), 30) # 这里的 30 是 Sigma 值，越大越糊
        blurred_frame = cv2.resize(heavy_blur, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # --- 新增功能：让右侧1/3的模糊稍微淡一点 ---
        # 1. 创建右侧1/3的掩码
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[:, w // 3 * 2:] = 255
        
        # 2. 创建一个较轻的模糊画面
        light_blur = cv2.GaussianBlur(small_frame, (25, 25), 10)
        light_blurred_frame = cv2.resize(light_blur, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # 3. 将较轻的模糊画面应用到掩码区域
        blurred_frame[mask == 255] = light_blurred_frame[mask == 255]
        
        display_frame = blurred_frame.copy() # 我们最终要在 display_frame 上作画显示

        # 3. 准备 MediaPipe 识别所需格式 (使用原图识别，确保精度)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        curr_time = time.time()

        try:
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                lms = results.multi_hand_landmarks[0]
                
                # --- 计算 Bounding Box 坐标 ---
                xs = [lm.x * w for lm in lms.landmark]
                ys = [lm.y * h for lm in lms.landmark]
                x1, y1 = max(0, int(min(xs))-20), max(0, int(min(ys))-20)
                x2, y2 = min(w, int(max(xs))+20), min(h, int(max(ys))+20)

                # --- 核心视觉特效：将手部框内的清晰图像覆盖到重度模糊背景上 ---
                display_frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2]

                # --- 绘制手部骨架 (画在最终显示的画面上) ---
                mp_draw.draw_landmarks(display_frame, lms, mp_hands.HAND_CONNECTIONS)

                # --- SVM 推理逻辑 ---
                row = normalize(lms.landmark)
                x_input = scaler.transform([row])
                probs = clf.predict_proba(x_input)[0]
                idx = probs.argmax()
                conf = probs[idx]
                char = clf.classes_[idx] if conf >= CONF_THRESHOLD else '?'

                # --- 状态检查：手势是否改变 ---
                if char != '?' and char == last_char:
                    stable_count += 1
                else:
                    stable_count = 0
                    last_char = char
                    action_fired = False # 手势改变，解锁

                # --- 核心交互逻辑 ---
                if stable_count >= STABLE_FRAMES and not action_fired:
                    if char == ' ':
                        if current_word:
                            word_history.append(current_word)
                            subtitle_text = current_word
                            subtitle_until = curr_time + 1.5
                            current_word = ''
                        action_fired = True
                    elif char == 'DEL_L':
                        if len(current_word) > 0: current_word = current_word[:-1]
                        action_fired = True
                    elif char == 'DEL_W':
                        current_word = ''
                        action_fired = True
                    elif char == '0':
                        word_history = []
                        action_fired = True
                    elif char == '6':
                        full_sentence = " ".join(word_history)
                        if current_word:
                            full_sentence = (full_sentence + " " + current_word).strip()
                        subtitle_text = full_sentence
                        subtitle_until = curr_time + 3.0 
                        action_fired = True
                    else:
                        current_word += char
                        action_fired = True

                # --- UI：绘制 Bounding Box 边框和识别文本 ---
                color = (0, 220, 0) if conf >= CONF_THRESHOLD else (0, 100, 255)
                if char == ' ': color = (255, 200, 0)      
                if char == '0': color = (0, 0, 255)        
                if char == '6': color = (255, 255, 0)      

                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                
                display_char = {' ': 'SPACE', '0': 'CLEAR', '6': 'SHOW'}.get(char, char)
                label = f'{display_char} ({conf:.0%})'
                
                cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4)
                cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            else:
                stable_count = 0
                last_char = None
                action_fired = False

        except Exception as e:
            cv2.putText(display_frame, f'Inference Error: {e}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            
        # --- UI：底部文字绘制 ---
        cv2.putText(display_frame, f'Word : {current_word}_', (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 4)
        cv2.putText(display_frame, f'Word : {current_word}_', (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        history_str = '  '.join(word_history[-5:])
        cv2.putText(display_frame, f'History: {history_str}', (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3)
        cv2.putText(display_frame, f'History: {history_str}', (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 255), 1)

        # --- UI：居中弹幕 ---
        if curr_time < subtitle_until and subtitle_text:
            (tw, th), _ = cv2.getTextSize(subtitle_text, cv2.FONT_HERSHEY_DUPLEX, 1.5, 3)
            tx = (w - tw) // 2
            cv2.putText(display_frame, subtitle_text, (tx, h - 150), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,0,0), 5)
            cv2.putText(display_frame, subtitle_text, (tx, h - 150), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 255), 2)

        # 最终显示的是处理过的 display_frame
        cv2.imshow(window_name, display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    for _ in range(5): cv2.waitKey(1)
    hands.close()

if __name__ == '__main__':
    main()