# 疲勞偵測（Signal + BPM）— 已訓練 Keras 模型

這個專案已包含訓練完成的 **Keras 完整模型**：`fatigue_prediction_model.keras`（結構＋權重）。  
模型以兩個特徵 **Signal、BPM** 預測 `Fatigue`（0=Normal, 1=Fatigue），可直接在本機推論，或匯出 **ONNX** 給 App 使用。

---

## 目錄（最小化）
```
├─ cleaned_train_data/        # 可選：清理後資料
├─ train_data/                # 可選：訓練/驗證彙整
├─ output_data/               # 可選：原始/中間輸出
├─ cleaned_data2.0.csv        # 範例資料（含 Signal, BPM, Status）
├─ fatigue_prediction_model.keras   # ✅ 已訓練模型
└─ （可選）scaler.joblib / model.meta.json
```

---

## 安裝（擇一）

### A. 在 Jupyter（推薦）
```python
%pip install -q numpy pandas scikit-learn matplotlib tensorflow tf2onnx onnxruntime
```

### B. 在終端機 / PowerShell
```bash
python -m pip install -U numpy pandas scikit-learn matplotlib tensorflow tf2onnx onnxruntime
```

---

## 快速驗證：檔案是否在正確位置
```python
import os
print("cwd =", os.getcwd())
print(os.listdir())  # 應能看到 fatigue_prediction_model.keras
```
> 看不到 `fatigue_prediction_model.keras`：請把 Notebook 的工作目錄切到模型所在資料夾，或改用**絕對路徑**（如 `r"C:\Users\User\Desktop\...\fatigue_prediction_model.keras"`）。

---

## 1) 單筆／小批量推論（載入 `.keras`）

```python
import os, numpy as np, pandas as pd, joblib, json
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

MODEL_PATH  = "fatigue_prediction_model.keras"  # 或絕對路徑
SCALER_PATH = "scaler.joblib"                   # 若已有就會載入；沒有就會臨時擬合

# 1) 載入模型
model = tf.keras.models.load_model(MODEL_PATH)

# 2) 準備標準化器（建議：直接載入已保存的 scaler）
if os.path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)
else:
    # 沒有 scaler 時，可用全資料臨時擬合（僅示範；正式環境請使用訓練時的 scaler）
    df_all = pd.read_csv("cleaned_data2.0.csv")
    scaler = StandardScaler().fit(df_all[["Signal","BPM"]].values)

# 3) 推論（Signal, BPM）
X_new = np.array([[520, 78],
                  [480, 62]], dtype=np.float32)
X_new_scaled = scaler.transform(X_new)
p = model.predict(X_new_scaled, verbose=0).ravel()  # 機率
pred = (p > 0.5).astype(int)                        # 0=Normal, 1=Fatigue
print("prob:", p, "pred:", pred)
```

---

## 2) 批次預測整個 CSV 並輸出結果檔

```python
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf, joblib, os

df = pd.read_csv("cleaned_data2.0.csv")
X = df[["Signal","BPM"]].values

# 載入或擬合 scaler
if os.path.exists("scaler.joblib"):
    scaler = joblib.load("scaler.joblib")
else:
    scaler = StandardScaler().fit(X)

X_scaled = scaler.transform(X)
model = tf.keras.models.load_model("fatigue_prediction_model.keras")
prob = model.predict(X_scaled, verbose=0).ravel()
pred = (prob > 0.5).astype(int)

out = df.copy()
out["FatigueProb"] = prob
out["FatiguePred"] = pred  # 0=Normal, 1=Fatigue
out.to_csv("predictions.csv", index=False)
print("Saved: predictions.csv")
```

---

## 3) 補存 `scaler.joblib` 與 `model.meta.json`（建議）

```python
import joblib, json
from sklearn.preprocessing import StandardScaler
import pandas as pd

df = pd.read_csv("cleaned_data2.0.csv")
scaler = StandardScaler().fit(df[["Signal","BPM"]].values)
joblib.dump(scaler, "scaler.joblib")

meta = {
  "features": ["Signal","BPM"],
  "target": "Fatigue (0=Normal, 1=Fatigue)",
  "threshold": 0.5,
  "model_file": "fatigue_prediction_model.keras",
  "scaler_file": "scaler.joblib"
}
open("model.meta.json","w",encoding="utf-8").write(json.dumps(meta,ensure_ascii=False,indent=2))
print("已保存：scaler.joblib, model.meta.json")
```

---

## 4) 匯出 ONNX（給 App / 跨語言）

```python
import tensorflow as tf, tf2onnx, onnxruntime as ort, numpy as np

model = tf.keras.models.load_model("fatigue_prediction_model.keras")
spec = (tf.TensorSpec((None, 2), tf.float32, name="x"),)  # 輸入兩個特徵
tf2onnx.convert.from_keras(model, input_signature=spec, opset=17,
                           output_path="fatigue_prediction_model.onnx")
print("已匯出 → fatigue_prediction_model.onnx")

# 簡單驗證 ONNX
sess = ort.InferenceSession("fatigue_prediction_model.onnx", providers=["CPUExecutionProvider"])
xb = np.array([[520,78],[480,62]], dtype=np.float32)
print("onnx logits/prob:", sess.run(None, {"x": xb})[0].ravel())
```

---

## 5)（可選）重訓並保存最佳權重 + scaler
若要用 `cleaned_data2.0.csv` 重新訓練後另存一份：

```python
import os, json, datetime, joblib, numpy as np, pandas as pd, tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

TS = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
OUT = os.path.join("models", f"fatigue_mlp-{TS}"); os.makedirs(OUT, exist_ok=True)

df = pd.read_csv("cleaned_data2.0.csv")
df["Fatigue"] = df["Status"].apply(lambda s: 0 if str(s).strip().lower()=="normal" else 1)
X = df[["Signal","BPM"]].values; y = df["Fatigue"].values

Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=0.2,random_state=42)
scaler = StandardScaler().fit(Xtr)
Xtr = scaler.transform(Xtr); Xte = scaler.transform(Xte)

model = Sequential([Input((2,)), Dense(64,activation="relu"), Dropout(0.3),
                    Dense(32,activation="relu"), Dense(1,activation="sigmoid")])
model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])

cbs = [ModelCheckpoint(os.path.join(OUT,"best.keras"), monitor="val_accuracy", save_best_only=True, verbose=1),
       EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True, verbose=1),
       CSVLogger(os.path.join(OUT,"train_log.csv"))]

model.fit(Xtr, ytr, epochs=50, batch_size=16, validation_data=(Xte,yte), callbacks=cbs, verbose=1)

model.save(os.path.join(OUT,"final.keras"))
joblib.dump(scaler, os.path.join(OUT,"scaler.joblib"))
open(os.path.join(OUT,"model.meta.json"),"w",encoding="utf-8").write(json.dumps(
  {"features":["Signal","BPM"],"threshold":0.5,"model_file":"final.keras","scaler_file":"scaler.joblib"}, ensure_ascii=False, indent=2))
print("Saved to:", OUT)
```

---

## 常見錯誤排查
- **`OSError: SavedModel file does not exist`**：模型路徑錯誤。用 `print(os.getcwd())` 與 `os.listdir()` 看看是否在同一層；必要時改用**絕對路徑**。  
- **`ModuleNotFoundError: No module named 'tensorflow'`**：尚未安裝，請執行上面的安裝指令。  
- **`ValueError: StandardScaler is not fitted`**：先依「補存 scaler」步驟建立 `scaler.joblib`，或在推論前以訓練資料擬合。  
- **ONNX 推論與 Keras 不一致**：確認輸入**標準化**一致、`opset=17`、輸入 shape `(None, 2)`。  

---

## 授權與用途
僅用於研究/健康輔助，非醫療診斷。請遵循資料隱私與在地法規。
