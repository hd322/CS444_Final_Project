"""
Phase 2: MediaPipe 关键点 + SVM 分类器
train_v2.py

用法:
    python train_v2.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ══════════════════════════════════════════
#  Config
# ══════════════════════════════════════════
DATA_PATH = './data/keypoints.csv'
SAVE_PATH = './checkpoint/svm_keypoints.pkl'

if __name__ == '__main__':
    # ── 加载数据 ──
    df = pd.read_csv(DATA_PATH)
    X  = df.drop('label', axis=1).values.astype(np.float32)
    y  = df['label'].values
    print(f'[数据] {len(df)} 条  |  类别: {sorted(set(y))}')

    # ── 划分训练/测试集 ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── 标准化 ──
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── 训练 SVM ──
    print('[训练] SVM (RBF kernel)...')
    clf = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
    clf.fit(X_train, y_train)

    # ── 评估 ──
    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f'\n[Test Acc] {acc:.4%}')
    print(classification_report(y_test, y_pred))

    # ── 保存模型 + scaler ──
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    with open(SAVE_PATH, 'wb') as f:
        pickle.dump({'clf': clf, 'scaler': scaler}, f)
    print(f'[保存] {SAVE_PATH}')
