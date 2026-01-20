import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import pickle
import numpy as np
import re
import math
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pyvi import ViTokenizer
from collections import defaultdict
import threading

MAX_LEN = 500
TRUNC_TYPE = "post"
W_CNN = 0.7
W_APRIORI = 0.3

vietnamese_stopwords = {
    "là", "của", "và", "những", "các", "trong", "đã", "cho", "người",
    "có", "được", "với", "không", "tại", "này", "để", "khi", "về", "như",
    "đang", "sẽ", "rất", "nhiều", "một", "theo", "từ", "đến"
}


def clean_and_segment_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = ViTokenizer.tokenize(text)
    words = text.split()
    return " ".join(w for w in words if w not in vietnamese_stopwords)


def normalize_text_vi(s: str) -> str:
    s = str(s).replace("\x00", " ")
    s = s.lower()
    s = re.sub(r"[^\w\sÀ-ỹ]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_keywords(text, vectorizer, k=30):
    text = normalize_text_vi(text)
    try:
        X = vectorizer.transform([text])
        vocab = np.array(vectorizer.get_feature_names_out())
        row = X.getrow(0)
        if row.nnz == 0: return []
        idx = row.indices
        vals = row.data
        top_idx = idx[np.argsort(-vals)[:k]]
        kws = vocab[top_idx].tolist()
        return [kw.replace(" ", "_") for kw in kws]
    except:
        return []


def build_rule_index(rules_df):
    rule_index = defaultdict(list)
    rules_store = []
    for idx, row in rules_df.iterrows():
        ant = row["antecedent"]
        rules_store.append({
            "id": idx, "antecedent": ant, "ant_len": len(ant),
            "label": row["label"], "confidence": row["confidence"], "support": row["support"]
        })
        for item in ant:
            rule_index[item].append(idx)
    return rule_index, rules_store


def predict_apriori(doc_keywords, rule_index, rules_store):
    candidate_counts = defaultdict(int)
    for kw in doc_keywords:
        if kw in rule_index:
            for rule_id in rule_index[kw]:
                candidate_counts[rule_id] += 1
    scores = defaultdict(float)
    for rule_id, count in candidate_counts.items():
        rule = rules_store[rule_id]
        if count == rule["ant_len"]:
            score = rule["confidence"] * (1.0 + 0.15 * math.log(1 + rule["ant_len"]))
            scores[rule["label"]] += score
    if not scores: return None, {}
    return max(scores, key=scores.get), dict(scores)


class ClassifierEngine:
    def __init__(self):
        self.is_ready = False
        self.load_error = None

    def load_models(self):
        try:
            # 1. Load CNN
            self.cnn_model = load_model("text_cnn_model.h5")
            with open("tokenizer.pkl", "rb") as f:
                self.tokenizer = pickle.load(f)
            with open("label_encoder.pkl", "rb") as f:
                self.le = pickle.load(f)

            # 2. Load Apriori
            with open("model_luat.pkl", "rb") as f:
                data = pickle.load(f)
                self.ap_vectorizer = data["vectorizer"]
                self.ap_k = data["params"]["k_keywords"]
                self.rule_index, self.rules_store = build_rule_index(data["rules"])

            self.is_ready = True
            print(">>> Đã tải xong toàn bộ Model!")
        except Exception as e:
            self.load_error = str(e)
            print(f"Lỗi tải model: {e}")

    def predict(self, text):
        if not self.is_ready:
            return "Chưa tải model", 0.0, "Error"

        processed_cnn = clean_and_segment_text(text)
        seq = self.tokenizer.texts_to_sequences([processed_cnn])
        padded = pad_sequences(seq, maxlen=MAX_LEN, truncating=TRUNC_TYPE)
        prob_cnn = self.cnn_model.predict(padded, verbose=0)[0]

        kws = extract_keywords(text, self.ap_vectorizer, k=self.ap_k)
        _, score_dict = predict_apriori(kws, self.rule_index, self.rules_store)

        prob_apriori = np.zeros_like(prob_cnn)
        if score_dict:
            total = sum(score_dict.values())
            for label, score in score_dict.items():
                try:
                    idx = self.le.transform([label])[0]
                    prob_apriori[idx] = score / total
                except:
                    pass

        # --- Weighted Voting ---
        if np.sum(prob_apriori) == 0:
            final_probs = prob_cnn
            source = "CNN (100%)"
        else:
            final_probs = (W_CNN * prob_cnn) + (W_APRIORI * prob_apriori)
            source = "Kết hợp (CNN + Apriori)"

        final_idx = np.argmax(final_probs)
        label_name = self.le.inverse_transform([final_idx])[0]
        confidence = final_probs[final_idx]

        return label_name, confidence, source


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ Thống Phân Loại Văn Bản Tự Động")
        self.root.geometry("800x600")

        self.engine = ClassifierEngine()

        lbl_title = tk.Label(root, text="NHẬP VĂN BẢN HOẶC TẢI FILE", font=("Arial", 14, "bold"), pady=10)
        lbl_title.pack()

        frame_btn = tk.Frame(root)
        frame_btn.pack(pady=5)

        btn_load_file = tk.Button(frame_btn, text="📂 Chọn File Text (.txt)", command=self.load_file, bg="#e1f5fe",
                                  font=("Arial", 10))
        btn_load_file.pack(side=tk.LEFT, padx=10)

        btn_clear = tk.Button(frame_btn, text="🗑️ Xóa Trắng", command=self.clear_text, font=("Arial", 10))
        btn_clear.pack(side=tk.LEFT, padx=10)

        self.txt_input = scrolledtext.ScrolledText(root, width=90, height=15, font=("Times New Roman", 12))
        self.txt_input.pack(pady=10)

        self.btn_predict = tk.Button(root, text="🔍 GÁN NHÃN NGAY", command=self.run_prediction,
                                     bg="#4caf50", fg="white", font=("Arial", 12, "bold"), height=2, width=20)
        self.btn_predict.pack(pady=10)

        frame_result = tk.LabelFrame(root, text="Kết Quả Dự Đoán", font=("Arial", 10, "italic"), padx=20, pady=10)
        frame_result.pack(fill="both", expand=True, padx=20, pady=10)

        self.lbl_result_label = tk.Label(frame_result, text="...", font=("Arial", 20, "bold"), fg="#d32f2f")
        self.lbl_result_label.pack()

        self.lbl_result_conf = tk.Label(frame_result, text="", font=("Arial", 11))
        self.lbl_result_conf.pack()

        self.status_bar = tk.Label(root, text="Đang tải model... Vui lòng đợi...", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        threading.Thread(target=self.init_engine, daemon=True).start()

    def init_engine(self):
        self.engine.load_models()
        if self.engine.is_ready:
            self.status_bar.config(text="Sẵn sàng! Model đã được tải thành công.", fg="green")
        else:
            self.status_bar.config(text=f"Lỗi tải model: {self.engine.load_error}. Kiểm tra lại thư mục.", fg="red")
            messagebox.showerror("Lỗi",
                                 "Không tìm thấy file model (h5/pkl).\nHãy đảm bảo chúng nằm cùng thư mục với file code này.")

    def load_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if not filepath:
            return
        try:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
            except:
                with open(filepath, "r", encoding="utf-16") as f:
                    content = f.read()

            self.txt_input.delete("1.0", tk.END)
            self.txt_input.insert(tk.END, content)
            self.status_bar.config(text=f"Đã tải file: {filepath}")
        except Exception as e:
            messagebox.showerror("Lỗi đọc file", str(e))

    def clear_text(self):
        self.txt_input.delete("1.0", tk.END)
        self.lbl_result_label.config(text="...")
        self.lbl_result_conf.config(text="")

    def run_prediction(self):
        if not self.engine.is_ready:
            messagebox.showwarning("Chưa sẵn sàng", "Model chưa tải xong hoặc bị lỗi. Vui lòng kiểm tra lại.")
            return

        text = self.txt_input.get("1.0", tk.END).strip()
        if len(text) < 5:
            messagebox.showinfo("Thông báo", "Vui lòng nhập văn bản dài hơn để dự đoán chính xác.")
            return

        try:
            label, conf, source = self.engine.predict(text)
            self.lbl_result_label.config(text=label.upper())
            self.lbl_result_conf.config(text=f"Độ tin cậy: {conf * 100:.2f}% | Nguồn: {source}")
        except Exception as e:
            messagebox.showerror("Lỗi Dự Đoán", str(e))


if __name__ == "__main__":
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    root = tk.Tk()
    app = App(root)
    root.mainloop()