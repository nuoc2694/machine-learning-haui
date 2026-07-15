import os
import re
import pickle
import numpy as np
import tensorflow as tf
from pyvi import ViTokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tqdm import tqdm

TRAIN_DIR = "Train_full"
TEST_DIR = "Test_full"

VOCAB_SIZE = 20000
EMBEDDING_DIM = 128
MAX_LEN = 500
TRUNC_TYPE = "post"
OOV_TOKEN = "<OOV>"

MODEL_PATH = "text_cnn_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"

vietnamese_stopwords = {
    "là", "của", "và", "những", "các", "trong", "đã", "cho", "người",
    "có", "được", "với", "không", "tại", "này", "để", "khi", "về", "như",
    "đang", "sẽ", "rất", "nhiều", "một", "theo", "từ", "đến"
}

def clean_and_segment_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = ViTokenizer.tokenize(text)
    words = text.split()
    return " ".join(w for w in words if w not in vietnamese_stopwords)

def load_data_from_directory(base_dir):
    texts, labels = [], []
    print(f"\nĐang đọc dữ liệu từ: {base_dir}")

    for label in os.listdir(base_dir):
        label_path = os.path.join(base_dir, label)
        if not os.path.isdir(label_path):
            continue

        files = [f for f in os.listdir(label_path) if f.endswith(".txt")]
        print(f"Nhãn '{label}' | {len(files)} file")

        for fname in tqdm(files, desc=f"   Xử lý {label}", unit="file"):
            fpath = os.path.join(label_path, fname)
            try:
                try:
                    with open(fpath, encoding="utf-8") as f:
                        content = f.read()
                except UnicodeDecodeError:
                    with open(fpath, encoding="utf-16") as f:
                        content = f.read()

                processed = clean_and_segment_text(content)
                if len(processed) > 10:
                    texts.append(processed)
                    labels.append(label)

            except Exception as e:
                print(f"Lỗi {fname}: {e}")

    return texts, labels

train_texts, train_labels = load_data_from_directory(TRAIN_DIR)
test_texts, test_labels = load_data_from_directory(TEST_DIR)

print("\nĐang mã hóa token...")
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
tokenizer.fit_on_texts(train_texts)

X_train = pad_sequences(
    tokenizer.texts_to_sequences(train_texts),
    maxlen=MAX_LEN,
    truncating=TRUNC_TYPE
)

X_test = pad_sequences(
    tokenizer.texts_to_sequences(test_texts),
    maxlen=MAX_LEN,
    truncating=TRUNC_TYPE
)

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_labels)
y_test = label_encoder.transform(test_labels)

print("\nĐang xây dựng mô hình...")
model = Sequential([
    Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),
    Conv1D(128, 5, activation="relu"),
    GlobalMaxPooling1D(),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(len(label_encoder.classes_), activation="softmax")
])

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()

print("\nĐang huấn luyện mô hình...")
early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

print("\nĐang đánh giá mô hình...")
y_pred = []

for i in tqdm(range(len(X_test)), desc="Predicting", unit="sample"):
    pred = model.predict(X_test[i:i+1], verbose=0)
    y_pred.append(np.argmax(pred))

y_pred = np.array(y_pred)

print("\nKết quả phân loại")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("\nĐang lưu mô hình")
model.save(MODEL_PATH)

with open(TOKENIZER_PATH, "wb") as f:
    pickle.dump(tokenizer, f)

with open(LABEL_ENCODER_PATH, "wb") as f:
    pickle.dump(label_encoder, f)

print("Lưu mô hình thành công")

def predict_text(text):
    cleaned = clean_and_segment_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, truncating=TRUNC_TYPE)
    pred = model.predict(padded, verbose=0)
    idx = np.argmax(pred[0])
    return label_encoder.inverse_transform([idx])[0], pred[0][idx]

sample_text = "Chính phủ vừa ban hành chính sách mới về y tế cộng đồng."
label, confidence = predict_text(sample_text)
print(f"\nDự đoán: {label} | Độ tin cậy: {confidence:.2f}")
