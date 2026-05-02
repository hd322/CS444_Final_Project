import cv2
import csv
import os
import mediapipe as mp
import time

# ══════════════════════════════════════════
#  Config
# ══════════════════════════════════════════
SAVE_PATH    = './data/keypoints.csv'
SAMPLE_RATE  = 3  
VALID_CHARS = set('ABCDEFGHIKLMNOPQRSTUVWXY 4')   # 末尾加空格
KEY_MAP = {'0': ' ', '4': '4'} 

# ══════════════════════════════════════════
#  MediaPipe 初始化
# ══════════════════════════════════════════
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

HEADER = ['label'] + [f'{axis}{i}' for i in range(21) for axis in ('x', 'y', 'z')]

def normalize(landmarks):
    wrist = landmarks[0]
    base  = landmarks[9]
    scale = ((base.x - wrist.x)**2 + (base.y - wrist.y)**2) ** 0.5
    scale = scale if scale > 1e-6 else 1.0
    row = []
    for lm in landmarks:
        row.extend([(lm.x - wrist.x) / scale, (lm.y - wrist.y) / scale, lm.z / scale])
    return row

def main():
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    file_exists = os.path.exists(SAVE_PATH)
    csvfile = open(SAVE_PATH, 'a', newline='')
    writer  = csv.writer(csvfile)
    if not file_exists:
        writer.writerow(HEADER)

    # ── 1. 极其保守的摄像头初始化 ──
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('[错误] 无法打开摄像头')
        return
    
    # 丢弃前 10 帧，给 macOS 摄像头和 MediaPipe 预热
    print("正在初始化摄像头和 AI 模型...")
    for _ in range(10):
        cap.read()
        time.sleep(0.01)

    print('[采集已启动]')
    print(' - 按字母键 (A-Y) 开始采集该类别')
    print(' - 按空格键 (Space) 停止采集（进入等待状态）')
    print(' - 按 Q 退出并保存')

    counts      = {}
    frame_count = 0
    active_char = None  # 当前正在录制的字母

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # ── 2. MediaPipe 处理 ──
        results = hands.process(rgb)
        
        is_saving = False
        if results.multi_hand_landmarks:
            lms = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)

            # 只有在指定了 active_char 时才保存
            if active_char and frame_count % SAMPLE_RATE == 0:
                row = normalize(lms.landmark)
                writer.writerow([active_char] + row)
                counts[active_char] = counts.get(active_char, 0) + 1
                is_saving = True

        frame_count += 1

        # ── 3. 统一的按键监听 (这很重要！) ──
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # 空格键停止
            active_char = None
        elif 0 <= key <= 127:
            char = chr(key).upper()
            char = KEY_MAP.get(chr(key), char)   # 加这行
            if char in VALID_CHARS:
                active_char = char

        # ── 4. UI 绘制 ──
        status = f'RECORDING: {active_char}' if active_char else 'IDLE (Press a key to start)'
        color  = (0, 255, 0) if is_saving else (0, 165, 255)
        
        # 阴影文字增加辨识度
        cv2.putText(frame, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4)
        cv2.putText(frame, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # 显示总数和右侧列表
        cv2.putText(frame, f'Total samples: {sum(counts.values())}', (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        y_offset = 30
        for ch in sorted(counts.keys()):
            cv2.putText(frame, f'{ch}: {counts[ch]}', (w - 100, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 20

        cv2.imshow('MediaPipe Collector', frame)

    # ── 5. 优雅收尾 ──
    csvfile.close()
    cap.release()
    cv2.destroyAllWindows()
    for _ in range(5): cv2.waitKey(1)
    hands.close()
    print('\n[采集结束] 数据已保存至:', SAVE_PATH)

if __name__ == '__main__':
    main()