import csv
import json
import pickle
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
import shutil
import subprocess

import cv2
import mediapipe as mp
import numpy as np

# ══════════════════════════════════════════
#  配置
# ══════════════════════════════════════════
MODEL_PATH = Path('./checkpoint/svm_keypoints.pkl')
CONF_THRESHOLD = 0.6
STABLE_FRAMES = 20
HOST = '127.0.0.1'
PORT = 8000

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


# ══════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════

def normalize(landmarks):
    wrist = landmarks[0]
    base = landmarks[9]
    scale = ((base.x - wrist.x) ** 2 + (base.y - wrist.y) ** 2) ** 0.5
    scale = scale if scale > 1e-6 else 1.0
    row = []
    for lm in landmarks:
        row.extend([(lm.x - wrist.x) / scale, (lm.y - wrist.y) / scale, lm.z / scale])
    return row


def display_char(char):
    return {' ': 'SPACE', '0': 'CLEAR', '6': 'SHOW'}.get(char, char)


HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Sign Reader Demo</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #111827;
      --muted: #6b7280;
      --line: #d8e0ea;
      --panel: #ffffff;
      --soft: #f5f8fc;
      --green: #18a76d;
      --blue: #2563eb;
      --amber: #d18616;
      --danger: #dc2626;
      --shadow: 0 18px 50px rgba(17, 24, 39, 0.14);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        linear-gradient(135deg, rgba(37, 99, 235, 0.08), rgba(24, 167, 109, 0.06)),
        #f8fafc;
    }

    .app {
      width: min(1500px, 100%);
      min-height: 100vh;
      margin: 0 auto;
      padding: 22px;
      display: grid;
      grid-template-rows: auto 1fr;
      gap: 16px;
    }

    header {
      display: grid;
      grid-template-columns: minmax(220px, 1fr) auto;
      gap: 16px;
      align-items: end;
    }

    h1 {
      margin: 0;
      font-size: clamp(28px, 4vw, 46px);
      line-height: 1;
    }

    .subtitle {
      margin: 8px 0 0;
      color: var(--muted);
      font-size: 15px;
    }

    .topbar {
      display: grid;
      grid-template-columns: auto auto auto;
      gap: 8px;
      align-items: center;
    }

    button {
      height: 38px;
      min-width: 38px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--ink);
      font: inherit;
      cursor: pointer;
      padding: 0 12px;
    }

    button:hover { background: var(--soft); border-color: #aab7c6; }
    button.primary { background: var(--blue); border-color: var(--blue); color: #fff; }
    button.danger { color: var(--danger); border-color: #fecaca; }
    button.active { background: #fef3c7; border-color: #f59e0b; color: #92400e; }

    main {
      display: grid;
      grid-template-columns: minmax(0, 1.3fr) minmax(380px, 0.7fr);
      gap: 16px;
      min-height: 0;
    }

    .video-shell,
    .side {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: var(--shadow);
      overflow: hidden;
    }

    .toolbar,
    .panel-head,
    .footer {
      min-height: 56px;
      padding: 12px 14px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      border-bottom: 1px solid var(--line);
      background: #fff;
    }

    .footer {
      border-bottom: 0;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: 13px;
      justify-content: space-between;
    }

    .toolbar strong, .panel-head h2 { font-size: 15px; margin: 0; }
    .metric, .muted { color: var(--muted); font-size: 13px; }

    .video-frame {
      background: #0b1220;
      min-height: 0;
    }

    .video-frame img {
      width: 100%;
      height: 100%;
      min-height: 460px;
      object-fit: cover;
      display: block;
    }

    .side {
      display: grid;
      grid-template-rows: auto auto 1fr auto;
      min-height: 520px;
    }

    .actions {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }

    .summary {
      padding: 14px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      border-bottom: 1px solid var(--line);
      background: #fbfdff;
    }

    .tile {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 12px;
    }

    .label {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
      text-transform: uppercase;
      font-weight: 700;
      letter-spacing: .02em;
    }

    .value {
      font-size: 24px;
      font-weight: 760;
      line-height: 1.1;
      overflow-wrap: anywhere;
    }

    .content {
      padding: 16px;
      display: grid;
      gap: 14px;
      min-height: 0;
    }

    .status {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: #344054;
    }

    .dot {
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: var(--green);
      box-shadow: 0 0 0 5px rgba(24, 167, 109, 0.14);
      flex: none;
    }

    .chat-log {
      min-height: 160px;
      padding: 16px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--soft);
      overflow: auto;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .empty-chat {
      margin: auto;
      color: var(--muted);
      font-size: 14px;
      text-align: center;
    }

    .message {
      max-width: 86%;
      align-self: flex-end;
      padding: 12px 14px;
      border-radius: 8px 8px 2px 8px;
      background: var(--blue);
      color: #fff;
      line-height: 1.35;
      overflow-wrap: anywhere;
      box-shadow: 0 10px 24px rgba(37, 99, 235, 0.18);
    }

    .composer {
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      display: grid;
      gap: 8px;
    }

    .draft-label {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }

    .draft {
      min-height: 58px;
      font-size: clamp(24px, 3vw, 34px);
      line-height: 1.18;
      overflow-wrap: anywhere;
    }

    .cursor {
      color: var(--blue);
      animation: blink 1s steps(2, start) infinite;
    }

    @keyframes blink { 50% { opacity: 0; } }

    .word-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      min-height: 32px;
    }

    .chip {
      padding: 7px 10px;
      border-radius: 8px;
      background: #e7f7ef;
      color: #09623f;
      font-weight: 650;
      font-size: 13px;
    }

    .progress-wrap {
      display: flex;
      align-items: center;
      gap: 12px;
      min-width: 240px;
    }

    .bar {
      width: 180px;
      height: 8px;
      border-radius: 999px;
      background: #dce5ee;
      overflow: hidden;
      flex: none;
    }

    .bar span {
      display: block;
      width: 0%;
      height: 100%;
      background: var(--green);
      transition: width 120ms linear;
    }

    @media (max-width: 1080px) {
      .app { padding: 16px; }
      header { grid-template-columns: 1fr; align-items: start; }
      main { grid-template-columns: 1fr; }
      .video-frame img { min-height: 330px; }
    }

    @media (max-width: 560px) {
      .summary { grid-template-columns: 1fr; }
      .footer { flex-direction: column; align-items: flex-start; }
      .progress-wrap { width: 100%; }
      .bar { width: 100%; }
    }
  </style>
</head>
<body>
  <div class="app">
    <header>
      <div>
        <h1>Sign Reader</h1>
        <p class="subtitle">Web demo of your original logic: stabilize predictions, build words, and show the full sentence on demand.</p>
      </div>
      <div class="topbar">
        <button id="toggleBlur">🔒 Privacy</button>
        <button id="backspace">⌫ Backspace</button>
        <button id="clear" class="danger">Clear</button>
        <button id="submit" class="primary">Enter / Show</button>
      </div>
    </header>

    <main>
      <section class="video-shell" aria-label="Video input">
        <div class="toolbar">
          <strong>Video Input</strong>
          <span class="metric" id="stableText">0 / 30 frames</span>
        </div>
        <div class="video-frame">
          <img src="/video_feed" alt="Camera video stream">
        </div>
      </section>

      <section class="side" aria-label="Controls and output">
        <div class="panel-head">
          <h2>Conversation</h2>
          <div class="actions">
            <button id="submit2" class="primary">↵</button>
            <button id="backspace2">⌫</button>
            <button id="clear2" class="danger">×</button>
          </div>
        </div>

        <div class="summary">
          <div class="tile">
            <span class="label">Current Sign</span>
            <span class="value" id="prediction">?</span>
          </div>
          <div class="tile">
            <span class="label">Confidence</span>
            <span class="value" id="confidence">0%</span>
          </div>
        </div>

        <div class="content">
          <div class="status"><span class="dot"></span><span id="status">Loading camera...</span></div>
          <div class="muted" id="modelStatus">Model status</div>

          <div class="chat-log" id="messages">
            <div class="empty-chat">Press Enter to send the current text.</div>
          </div>

          <div class="composer">
            <span class="draft-label">Current Text</span>
            <div class="draft" id="sentence"><span class="cursor">_</span></div>
          </div>

          <div class="word-row" id="history"></div>
        </div>

        <div class="footer">
          <span id="footerText">Stable trigger rules: SPACE commits a word, DEL_L deletes one char, DEL_W clears current word.</span>
          <div class="progress-wrap">
            <span class="metric" id="progressText">0%</span>
            <div class="bar" aria-hidden="true"><span id="progress"></span></div>
          </div>
        </div>
      </section>
    </main>
  </div>

  <script>
    const $ = (id) => document.getElementById(id);

    async function post(path) {
      await fetch(path, { method: 'POST' });
      await refresh();
    }

    function renderWords(words) {
      $('history').innerHTML = '';
      (words || []).slice(-10).forEach((word) => {
        const chip = document.createElement('span');
        chip.className = 'chip';
        chip.textContent = word;
        $('history').appendChild(chip);
      });
    }

    function renderMessages(messages) {
      $('messages').innerHTML = '';
      if (!messages || !messages.length) {
        const empty = document.createElement('div');
        empty.className = 'empty-chat';
        empty.textContent = 'Press Enter to send the current text.';
        $('messages').appendChild(empty);
        return;
      }
      messages.forEach((text) => {
        const bubble = document.createElement('div');
        bubble.className = 'message';
        bubble.textContent = text;
        $('messages').appendChild(bubble);
      });
      $('messages').scrollTop = $('messages').scrollHeight;
    }

    async function refresh() {
      const res = await fetch('/api/state', { cache: 'no-store' });
      const state = await res.json();
      const sentence = state.sentence || '';

      $('sentence').innerHTML = '';
      $('sentence').append(document.createTextNode(sentence));
      const cursor = document.createElement('span');
      cursor.className = 'cursor';
      cursor.textContent = '_';
      $('sentence').appendChild(cursor);

      $('prediction').textContent = state.prediction || '?';
      $('confidence').textContent = Math.round((state.confidence || 0) * 100) + '%';
      $('status').textContent = state.status || 'Ready';
      $('stableText').textContent = `${state.stableFrames} / ${state.stableTarget} frames`;
      const progress = Math.min(100, (state.stableFrames || 0) / Math.max(1, state.stableTarget || 30) * 100);
      $('progress').style.width = `${progress}%`;
      $('progressText').textContent = `${Math.round(progress)}%`;
      $('modelStatus').textContent = state.hasModel ? `Model ready | classes: ${state.classes.join(', ')}` : 'No model loaded';
      $('toggleBlur').classList.toggle('active', state.blurBackground); 
      $('toggleBlur').textContent = state.blurBackground ? '🔓 Unblur' : '🔒 Privacy';  
      renderWords(state.history || []);
      renderMessages(state.messages || []);
    }

    $('submit').addEventListener('click', () => post('/api/submit'));
    $('submit2').addEventListener('click', () => post('/api/submit'));
    $('backspace').addEventListener('click', () => post('/api/backspace'));
    $('backspace2').addEventListener('click', () => post('/api/backspace'));
    $('clear').addEventListener('click', () => post('/api/clear'));
    $('clear2').addEventListener('click', () => post('/api/clear'));
    $('toggleBlur').addEventListener('click', () => post('/api/toggle_blur'));

    window.addEventListener('keydown', (event) => {
      if (event.target.tagName === 'INPUT') return;
      if (event.key === 'Backspace' || event.key === 'Delete') {
        event.preventDefault();
        post('/api/backspace');
      } else if (event.key === 'Enter') {
        event.preventDefault();
        post('/api/submit');
      }
    });

    refresh();
    setInterval(refresh, 500);
  </script>
</body>
</html>
'''


# ══════════════════════════════════════════
#  核心 demo
# ══════════════════════════════════════════
class SignLanguageWebDemo:
    def __init__(self):
        try:
            with open(MODEL_PATH, 'rb') as f:
                bundle = pickle.load(f)
            self.clf = bundle['clf']
            self.scaler = bundle['scaler']
            print(f'[Model] Loaded successfully. Classes: {self.clf.classes_}')
        except Exception as e:
            raise RuntimeError(f'Can not load model weight: {e}')

        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError('Can not open camera')

        self.lock = threading.Lock()
        self.frame_lock = threading.Lock()
        self.current_word = ''
        self.word_history = []
        self.messages = []
        self.last_char = None
        self.stable_count = 0
        self.action_fired = False
        self.prediction = '?'
        self.confidence = 0.0
        self.status = 'Waiting for hand'
        self.subtitle_text = ''
        self.subtitle_until = 0.0
        self.frame_count = 0
        self.blur_background = False

        print('Warming up...')
        for _ in range(15):
            self.cap.read()
            time.sleep(0.01)
        print('[Demo] Ready. Open http://%s:%s in your browser' % (HOST, PORT))

    def snapshot(self):
        with self.lock:
            sentence_parts = self.word_history + ([self.current_word] if self.current_word else [])
            return {
                'currentWord': self.current_word,
                'history': list(self.word_history),
                'messages': list(self.messages),
                'sentence': ' '.join(sentence_parts).strip(),
                'prediction': display_char(self.prediction),
                'confidence': round(float(self.confidence), 3),
                'stableFrames': int(self.stable_count),
                'stableTarget': STABLE_FRAMES,
                'status': self.status,
                'hasModel': True,
                'classes': [display_char(ch) for ch in self.clf.classes_],
                'blurBackground': self.blur_background,
            }

    def _reset_motion_locked(self):
        self.stable_count = 0
        self.last_char = None
        self.action_fired = False

    def backspace(self):
        with self.lock:
            if self.current_word:
                self.current_word = self.current_word[:-1]
            elif self.word_history:
                self.current_word = self.word_history.pop()[:-1]
            self._reset_motion_locked()
            self.status = 'Deleted character'

    def toggle_blur(self):
        with self.lock:
            self.blur_background = not self.blur_background
            return self.blur_background

    def clear(self):
        with self.lock:
            self.current_word = ''
            self.word_history = []
            self.messages = []
            self.subtitle_text = ''
            self.subtitle_until = 0.0
            self._reset_motion_locked()
            self.status = 'Cleared'

    def speak(self, sentence):
        if not shutil.which('say'):
            return
        def run_speech():
            try:
                # subprocess.run(['say', sentence], check=False)
                subprocess.run(['say', '-v', 'Samantha', sentence], check=False)
            except OSError:
                pass
        threading.Thread(target=run_speech, daemon=True).start()

    def submit(self):
        with self.lock:
            sentence_parts = self.word_history + ([self.current_word] if self.current_word else [])
            sentence = ' '.join(sentence_parts).strip()
            if sentence:
                self.messages.append(sentence)
                self.subtitle_text = sentence
                self.subtitle_until = time.time() + 3.0
            self.current_word = ''
            self.word_history = []
            self._reset_motion_locked()
            self.status = 'Sent text' if sentence else 'Nothing to send'

            if sentence: self.speak(sentence)

    def _handle_prediction_locked(self, char, conf):
        now = time.time()
        self.prediction = char
        self.confidence = float(conf)
        sentence_to_speak = ''     

        if char != '?' and char == self.last_char:
            self.stable_count += 1
        else:
            self.stable_count = 0
            self.last_char = char
            self.action_fired = False

        if self.stable_count < STABLE_FRAMES or self.action_fired:
            return sentence_to_speak

        if char == ' ':
            if self.current_word:
                self.word_history.append(self.current_word)
                self.subtitle_text = self.current_word
                self.subtitle_until = now + 1.5
                self.current_word = ''
            self.action_fired = True
            self.status = 'SPACE committed'
        elif char == 'DEL_L':
            if self.current_word:
                self.current_word = self.current_word[:-1]
            self.action_fired = True
            self.status = 'Deleted last letter'
        elif char == 'DEL_W':
            self.current_word = ''
            self.action_fired = True
            self.status = 'Deleted current word'
        elif char == '0':
            self.word_history = []
            self.action_fired = True
            self.status = 'History cleared'
        elif char == '6':
            full_sentence = ' '.join(self.word_history)
            if self.current_word:
                full_sentence = (full_sentence + ' ' + self.current_word).strip()
            self.subtitle_text = full_sentence
            self.subtitle_until = now + 3.0
            self.action_fired = True
            self.status = 'Showing full sentence'
            sentence_to_speak = full_sentence 
        else:
            self.current_word += char
            self.action_fired = True
            self.status = f'Added {char}'
        return sentence_to_speak

    def _process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        """
        Add for gauss blur for privacy
        """
        # display_frame = frame.copy()
        with self.lock:
            blur_bg = self.blur_background

        if blur_bg:
            small = cv2.resize(frame, (max(1, w // 4), max(1, h // 4)))
            heavy_blur = cv2.GaussianBlur(small, (99, 99), 12)
            display_frame = cv2.resize(heavy_blur, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            display_frame = frame.copy()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.frame_count += 1

        try:
            results = self.hands.process(rgb)

            if results.multi_hand_landmarks:
                lms = results.multi_hand_landmarks[0]

                xs = [lm.x * w for lm in lms.landmark]
                ys = [lm.y * h for lm in lms.landmark]
                x1, y1 = max(0, int(min(xs)) - 20), max(0, int(min(ys)) - 20)
                x2, y2 = min(w, int(max(xs)) + 20), min(h, int(max(ys)) + 20)

                if blur_bg:
                    display_frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2]

                mp_draw.draw_landmarks(display_frame, lms, mp_hands.HAND_CONNECTIONS)

                row = normalize(lms.landmark)
                x_input = self.scaler.transform([row])
                probs = self.clf.predict_proba(x_input)[0]
                idx = probs.argmax()
                conf = float(probs[idx])
                char = self.clf.classes_[idx] if conf >= CONF_THRESHOLD else '?'

                with self.lock:
                    sentence_to_speak = self._handle_prediction_locked(char, conf)  # ← 接收返回值

                if sentence_to_speak:
                    self.speak(sentence_to_speak)  # ← 锁外调用

                # 边框颜色
                color = (0, 220, 0) if conf >= CONF_THRESHOLD else (0, 100, 255)
                if char == ' ': color = (255, 200, 0)
                if char == '0': color = (0, 0, 255)
                if char == '6': color = (255, 255, 0)
                if char == 'DEL_L': color = (255, 120, 0)
                if char == 'DEL_W': color = (255, 0, 255)

                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 3)

                # 字体更清晰：白底描边 + 彩色主体，字号稍大
                label = f'{display_char(char)}  {conf:.0%}'
                cv2.putText(display_frame, label, (x1, max(36, y1 - 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (8, 12, 20), 6)
                cv2.putText(display_frame, label, (x1, max(36, y1 - 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
            else:
                with self.lock:
                    self.status = 'Waiting for hand'
                    self.stable_count = 0
                    self.last_char = None
                    self.action_fired = False
                    self.prediction = '?'
                    self.confidence = 0.0
        except Exception as e:
            with self.lock:
                self.status = f'Inference error: {e}'
            cv2.putText(display_frame, 'Inference Error', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 底部文字
        with self.lock:
            current_word = self.current_word
            word_history = list(self.word_history)
            subtitle_text = self.subtitle_text
            subtitle_until = self.subtitle_until

        cv2.putText(display_frame, f'Word : {current_word}_', (10, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
        cv2.putText(display_frame, f'Word : {current_word}_', (10, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        history_str = '  '.join(word_history[-5:])
        cv2.putText(display_frame, f'History: {history_str}', (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
        cv2.putText(display_frame, f'History: {history_str}', (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 255), 1)

        if subtitle_text and time.time() < subtitle_until:
            (tw, th), _ = cv2.getTextSize(subtitle_text, cv2.FONT_HERSHEY_DUPLEX, 1.5, 3)
            tx = max(0, (w - tw) // 2)
            cv2.putText(display_frame, subtitle_text, (tx, h - 150),
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 5)
            cv2.putText(display_frame, subtitle_text, (tx, h - 150),
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 255), 2)

        # 顶部状态面板
        cv2.rectangle(display_frame, (0, 0), (w, 112), (10, 14, 22), -1)
        with self.lock:
            status = self.status
            prediction = display_char(self.prediction)
            conf = self.confidence
            progress = min(1.0, self.stable_count / max(1, STABLE_FRAMES))
        cv2.putText(display_frame, f'Status: {status}', (22, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
        cv2.putText(display_frame, f'Predict: {prediction} | Conf: {conf:.0%}', (22, 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
        cv2.rectangle(display_frame, (22, 76), (222, 84), (45, 53, 67), -1)
        cv2.rectangle(display_frame, (22, 76), (22 + int(200 * progress), 84), (56, 214, 132), -1)

        return display_frame

    def frames(self):
        while True:
            with self.frame_lock:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    time.sleep(0.03)
                    continue
                frame = self._process_frame(frame)

            ok, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if not ok:
                continue

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
            )

    def close(self):
        try:
            self.cap.release()
        finally:
            self.hands.close()


# ══════════════════════════════════════════
#  HTTP 服务
# ══════════════════════════════════════════
class DemoRequestHandler(BaseHTTPRequestHandler):
    demo = None

    def log_message(self, format, *args):
        return

    def _send_bytes(self, body, content_type, status=HTTPStatus.OK):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload, status=HTTPStatus.OK):
        self._send_bytes(
            json.dumps(payload, ensure_ascii=False).encode('utf-8'),
            'application/json; charset=utf-8',
            status,
        )

    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/':
            self._send_bytes(HTML.encode('utf-8'), 'text/html; charset=utf-8')
        elif path == '/api/state':
            self._send_json(self.demo.snapshot())
        elif path == '/video_feed':
            self.send_response(HTTPStatus.OK)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            try:
                for frame in self.demo.frames():
                    self.wfile.write(frame)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self._send_bytes(b'Not found', 'text/plain; charset=utf-8', HTTPStatus.NOT_FOUND)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            if path == '/api/backspace':
                self.demo.backspace()
            elif path == '/api/clear':
                self.demo.clear()
            elif path == '/api/submit':
                self.demo.submit()
            elif path == '/api/toggle_blur':
                self.demo.toggle_blur()
            else:
                self._send_bytes(b'Not found', 'text/plain; charset=utf-8', HTTPStatus.NOT_FOUND)
                return
            self._send_json({'ok': True})
        except Exception as exc:
            self._send_json({'ok': False, 'error': str(exc)}, HTTPStatus.BAD_REQUEST)


def main():
    try:
        demo = SignLanguageWebDemo()
    except Exception as exc:
        print(f'[Error] {exc}')
        return

    DemoRequestHandler.demo = demo
    server = ThreadingHTTPServer((HOST, PORT), DemoRequestHandler)
    state = demo.snapshot()
    print(f"[Model] classes: {state['classes']}")
    print(f'[Demo] Open http://{HOST}:{PORT} in your browser')
    print('[Demo] Press Ctrl+C here to quit')

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        demo.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
