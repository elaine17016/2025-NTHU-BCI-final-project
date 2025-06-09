import numpy as np
import h5py
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# === 載入資料 ===
file = h5py.File('MI_2_7-30hz.mat', 'r')  # 替換成你的 .mat 檔案
X = np.array(file['X'])  # 原 shape: (576, 3, 500)
X = X.transpose(2, 1, 0)  # → (576, 500, 3)
X = X.reshape(-1, 500)  # (576 × 3, 500) → (1728, 500)

y = np.array(file['y']).flatten()  # (576,)
# 標籤也要複製三次，讓每個 channel 對應正確 label
y = np.repeat(y, 3)  # 576 → 1728
print(f'✅ X shape: {X.shape}, y shape: {y.shape}')

# === 拆分資料：80% 訓練+驗證，20% 測試 ===
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === Cross-validation 設定 ===
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_acc_list = []
val_acc_list = []
class_correct = defaultdict(int)
class_total = defaultdict(int)

fold = 1
for train_idx, val_idx in skf.split(X_trainval, y_trainval):
    X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
    y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]

    # 特徵標準化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # 建立 XGBoost 模型
    clf = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        eval_metric='logloss',
        random_state=42
    )
    clf.fit(X_train, y_train)

    y_train_pred = clf.predict(X_train)
    y_val_pred = clf.predict(X_val)

    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)

    print(f'📘 Fold {fold}: Train Acc = {train_acc:.4f} | Val Acc = {val_acc:.4f}')
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
    fold += 1

    # 每類別的驗證正確率
    for true, pred in zip(y_val, y_val_pred):
        class_total[true] += 1
        if true == pred:
            class_correct[true] += 1

# === 在全部 trainval 上重新訓練一次 ===
scaler_final = StandardScaler()
X_trainval_scaled = scaler_final.fit_transform(X_trainval)
X_test_scaled = scaler_final.transform(X_test)

clf_final = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
clf_final.fit(X_trainval_scaled, y_trainval)

y_test_pred = clf_final.predict(X_test_scaled)
test_acc = accuracy_score(y_test, y_test_pred)

# === 輸出結果 ===
print("\n✅ Cross-Validation Summary")
print(f'📊 平均 Train Acc: {np.mean(train_acc_list):.4f}')
print(f'🧪 平均 Val Acc: {np.mean(val_acc_list):.4f}')

print("\n🎯 每類別驗證集正確率：")
label_names = ['Left Hand', 'Right Hand']
for i in [0, 1]:
    acc = class_correct[i] / class_total[i] if class_total[i] > 0 else 0
    print(f'📡 {label_names[i]} Accuracy: {acc:.4f} ({class_correct[i]}/{class_total[i]})')

print(f'\n🚀 測試集準確率: {test_acc:.4f}')
print("\n📋 測試集分類報告:")
print(classification_report(y_test, y_test_pred, target_names=label_names))

# =========================
# 以下為新增視覺化程式碼
# =========================

# 1. 訓練與驗證準確率折線圖
plt.figure(figsize=(8, 5))
plt.plot(range(1, 6), train_acc_list, marker='o', label='Training Accuracy')
plt.plot(range(1, 6), val_acc_list, marker='s', label='Validation Accuracy')
plt.title('Cross-Validation Accuracy per Fold')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.xticks(range(1, 6))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. 混淆矩陣 (Confusion Matrix)
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
plt.title('Confusion Matrix on Test Set')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

# 3. 每類別驗證準確率長條圖
val_class_acc = [class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in [0, 1]]

plt.figure(figsize=(6, 4))
sns.barplot(x=label_names, y=val_class_acc)
plt.title('Per-Class Validation Accuracy')
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.tight_layout()
plt.show()

# 4. 測試集分類報告圖（Precision / Recall / F1）
report = classification_report(y_test, y_test_pred, target_names=label_names, output_dict=True)
metrics = ['precision', 'recall', 'f1-score']

plt.figure(figsize=(10, 4))
for i, metric in enumerate(metrics):
    plt.subplot(1, 3, i+1)
    scores = [report[cls][metric] for cls in label_names]
    sns.barplot(x=label_names, y=scores)
    plt.title(metric.capitalize())
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()

plt.suptitle("Classification Report Metrics (Test Set)", fontsize=14, y=1.05)
plt.tight_layout()
plt.show()

# 5. PCA 二維視覺化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_test_scaled)

plt.figure(figsize=(6, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_test_pred, style=y_test, palette='Set2')
plt.title('PCA Projection of Test Set')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend(title='Predicted / True', loc='best')
plt.tight_layout()
plt.show()