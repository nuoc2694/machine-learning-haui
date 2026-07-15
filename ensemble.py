import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import defaultdict
import math
import re
from pyvi import ViTokenizer

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

def extract_keywords_for_new_doc(text, vectorizer, k=30):
    text = normalize_text_vi(text)
    X = vectorizer.transform([text])
    vocab = np.array(vectorizer.get_feature_names_out())
    row = X.getrow(0)
    if row.nnz == 0: return []
    idx = row.indices
    vals = row.data
    top_idx = idx[np.argsort(-vals)[:k]]
    kws = vocab[top_idx].tolist()
    return [kw.replace(" ", "_") for kw in kws]

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

def predict_label_fast(doc_keywords, rule_index, rules_store):
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
    if not scores: return None, [], {}
    pred = max(scores.items(), key=lambda x: x[1])[0]
    return pred, [], dict(scores)


print("Đang tải các model từ ổ cứng...")

cnn_model = load_model("text_cnn_model.h5")
with open("tokenizer.pkl", "rb") as f:
    cnn_tokenizer = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("model_luat.pkl", "rb") as f:
    apriori_model_data = pickle.load(f)

apriori_vectorizer = apriori_model_data["vectorizer"]
apriori_rules = apriori_model_data["rules"]
apriori_k = apriori_model_data["params"]["k_keywords"]
rule_index, rules_store = build_rule_index(apriori_rules)

print(">>> Đã tải xong! Sẵn sàng dự đoán.")


def predict_ensemble(text_input):
    cnn_text = clean_and_segment_text(text_input)
    seq = cnn_tokenizer.texts_to_sequences([cnn_text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, truncating=TRUNC_TYPE)
    prob_cnn = cnn_model.predict(padded, verbose=0)[0]

    kws = extract_keywords_for_new_doc(text_input, apriori_vectorizer, k=apriori_k)
    _, _, score_dict = predict_label_fast(kws, rule_index, rules_store)

    prob_apriori = np.zeros_like(prob_cnn)
    if score_dict:
        total_score = sum(score_dict.values())
        for label_name, score in score_dict.items():
            try:
                idx = label_encoder.transform([label_name])[0]
                prob_apriori[idx] = score / total_score
            except ValueError:
                pass

    if np.sum(prob_apriori) == 0:
        final_probs = prob_cnn
    else:
        final_probs = (W_CNN * prob_cnn) + (W_APRIORI * prob_apriori)

    final_idx = np.argmax(final_probs)
    final_label = label_encoder.inverse_transform([final_idx])[0]

    return final_label, final_probs[final_idx]

new_text = "Cầu thủ ghi bàn thắng quyết định vào lưới đối phương trong trận chung kết"
ket_qua, do_tin_cay = predict_ensemble(new_text)

print(f"\nCâu: {new_text}")
print(f"Dự đoán cuối cùng: {ket_qua}")
print(f"Độ tin cậy: {do_tin_cay:.4f}")
