import numpy as np
import h5py
from scipy.io import loadmat
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

file = h5py.File('MI_2_7-30hz.mat', 'r')
X_raw = file['X']
print("🔍 原始 X shape:", X_raw.shape)
# 如果你確定 shape 是 (500, 3, 576)，那可以這樣轉換：
X = X_raw[:].transpose(2, 1, 0)  # 得到 (576, 3, 500)
print("✅ 轉置後 X shape:", X.shape)

y_raw = file['y']
print("🔍 原始 y shape:", y_raw.shape)
# 如果你確定 shape 是 (1, 576)，那可以這樣轉換：
y = y_raw[:].flatten()  # 得到 (576, 1)
print("✅ 轉置後 y shape:", y.shape)
# === 載入 .mat 檔 ===
# mat = loadmat('MI_2.mat')  # ← 修改成你的檔名
# X = mat['X']  # shape: (576, 3, 500)
# y = mat['y'].flatten()  # shape: (576,)

print(f'✅ 資料維度: {X.shape}, 標籤維度: {y.shape}')
################################################
# 如果想要讓資料變成(576,1500)的格式的話請用這邊，這邊的好處是保留了channel的資訊
# 因為MI會在腦袋的另一邊產生desynchronization，所以保留channel應該有其必要性
# 但是這樣的缺點是資料筆數太少576而feature太多1500，我自己train A02T的test acc只有0.5
# === reshape: (576, 3, 500) → (576, 1500) ===
# X = X.reshape((X.shape[0], -1))  # (576, 1500)
################################################

################################################
# # 如果想要讓資料變成(1728,500)的格式的話請用這邊，這邊的好處是增加了資料筆數但是channel的資訊就不見了
# # 我自己train A02T的test acc只有0.8931
# # 原始資料：X (576, 3, 500)，y (576,)
# # 改成每個 channel 當作獨立樣本
X = X.reshape(-1, 500)  # (576 × 3, 500) → (1728, 500)

# 標籤也要複製三次，讓每個 channel 對應正確 label
y = np.repeat(y, 3)  # 576 → 1728
################################################

print(f'✅ 轉換後資料維度: X={X.shape}, y={y.shape}')

# === 拆分資料: 80% 訓練驗證, 20% 測試 ===
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f'📦 訓練驗證資料: {X_trainval.shape}, 測試資料: {X_test.shape}')

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

    # 標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 訓練 SVM
    clf = SVC(kernel='rbf', C=1, gamma='scale')
    clf.fit(X_train_scaled, y_train)

    # 預測
    y_train_pred = clf.predict(X_train_scaled)
    y_val_pred = clf.predict(X_val_scaled)

    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)

    print(f'📘 Fold {fold}: Training Acc = {train_acc:.4f} | Validation Acc = {val_acc:.4f}')
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
    fold += 1

    for true, pred in zip(y_val, y_val_pred):
        true = int(true)
        pred = int(pred)
        class_total[true] += 1
        if true == pred:
            class_correct[true] += 1

# === 在全部訓練驗證資料上重新訓練模型 ===
scaler_final = StandardScaler()
X_trainval_scaled = scaler_final.fit_transform(X_trainval)
X_test_scaled = scaler_final.transform(X_test)

clf_final = SVC(kernel='rbf', C=1, gamma='scale')
clf_final.fit(X_trainval_scaled, y_trainval)

y_test_pred = clf_final.predict(X_test_scaled)
test_acc = accuracy_score(y_test, y_test_pred)

# === 分析結果 ===
print("\n✅ Cross-Validation 結果")
print(f'📊 平均 Training Acc: {np.mean(train_acc_list):.4f}')
print(f'🧪 平均 Validation Acc: {np.mean(val_acc_list):.4f}')

print("\n🎯 每類別在驗證集的準確率：")
label_names = ['Left Hand', 'Right Hand']
for i in [0, 1]:
    acc = class_correct[i] / class_total[i] if class_total[i] > 0 else 0
    print(f'📡 {label_names[i]} Accuracy: {acc:.4f} ({class_correct[i]}/{class_total[i]})')

print(f'\n🚀 最終測試集準確率: {test_acc:.4f}')
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
