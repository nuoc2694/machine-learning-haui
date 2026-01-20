import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
import re
import glob
import math
from collections import defaultdict
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

def build_rule_index(rules_df):
    rule_index = defaultdict(list)
    rules_store = []

    for idx, row in rules_df.iterrows():
        ant = row["antecedent"]
        rules_store.append({
            "id": idx,
            "antecedent": ant,
            "ant_len": len(ant),
            "label": row["label"],
            "confidence": row["confidence"],
            "support": row["support"]
        })
        for item in ant:
            rule_index[item].append(idx)

    return rule_index, rules_store


def read_text_smart(path: str) -> str:
    encodings = ["utf-16", "utf-16le", "utf-8-sig", "utf-8", "windows-1258", "cp1252", "latin-1"]
    raw = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                raw = f.read()
            break
        except Exception:
            continue

    if raw is None:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()

    return raw


def normalize_text_vi(s: str) -> str:
    s = s.replace("\x00", " ")
    s = s.lower()
    s = re.sub(r"[^\w\sÀ-ỹ]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_dataset_from_folders(root_dir: str):
    texts, labels, paths = [], [], []

    for label_dir in sorted(os.listdir(root_dir)):
        full_dir = os.path.join(root_dir, label_dir)
        if not os.path.isdir(full_dir):
            continue

        files = glob.glob(os.path.join(full_dir, "*.txt"))
        for fp in tqdm(files, desc=f"Đọc dữ liệu [{label_dir}]"):
            raw = read_text_smart(fp)
            txt = normalize_text_vi(raw)
            if len(txt) == 0:
                continue
            texts.append(txt)
            labels.append(label_dir.upper())
            paths.append(fp)

    return pd.DataFrame({"path": paths, "text": texts, "label": labels})


def top_k_keywords_per_doc_tfidf(
    texts,
    k=30,
    min_df=2,
    max_df=0.9,
    ngram_range=(1, 2),
    stopwords_vi=None
):
    vectorizer = TfidfVectorizer(
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        stop_words=stopwords_vi,
        token_pattern=r"(?u)\b\w+\b"
    )
    X = vectorizer.fit_transform(texts)
    vocab = np.array(vectorizer.get_feature_names_out())

    keywords_list = []
    for i in tqdm(range(X.shape[0]), desc="Trích xuất TF‑IDF"):
        row = X.getrow(i)
        if row.nnz == 0:
            keywords_list.append([])
            continue
        idx = row.indices
        vals = row.data
        top_idx = idx[np.argsort(-vals)[:k]]

        kws = vocab[top_idx].tolist()
        kws = [kw.replace(" ", "_") for kw in kws]
        keywords_list.append(kws)

    return keywords_list, vectorizer


def build_transactions(keywords_list, labels):
    return [kw + [labels[i]] for i, kw in enumerate(keywords_list)]


def mine_frequent_itemsets(transactions, min_support=0.02, max_len=4):
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    fi = apriori(df, min_support=min_support, use_colnames=True, max_len=max_len)
    fi["length"] = fi["itemsets"].apply(len)
    return fi, df.columns.tolist()


def generate_label_rules(frequent_itemsets: pd.DataFrame, all_items, labels_set, min_conf=0.6):
    support_map = {frozenset(s): sup for s, sup in zip(frequent_itemsets["itemsets"], frequent_itemsets["support"])}

    rules = []
    for itemset, sup_AL in tqdm(
        support_map.items(),
        total=len(support_map),
        desc="Sinh luật kết hợp"
    ):
        labels_in_set = [x for x in itemset if x in labels_set]
        if len(labels_in_set) != 1:
            continue

        L = labels_in_set[0]
        A = frozenset([x for x in itemset if x != L])
        if len(A) == 0:
            continue

        sup_A = support_map.get(A)
        if sup_A is None or sup_A == 0:
            continue

        conf = sup_AL / sup_A
        if conf >= min_conf:
            rules.append({
                "antecedent": tuple(sorted(A)),
                "label": L,
                "support": sup_AL,
                "confidence": conf,
                "antecedent_support": sup_A,
                "lift": conf / support_map.get(frozenset([L]), 1e-12)
            })

    rules = pd.DataFrame(rules).sort_values(["confidence", "support", "lift"], ascending=False)
    return rules


def predict_label_fast(doc_keywords, rule_index, rules_store, topn_match=50):
    kw_set = set(doc_keywords)

    candidate_counts = defaultdict(int)
    for kw in doc_keywords:
        if kw in rule_index:
            for rule_id in rule_index[kw]:
                candidate_counts[rule_id] += 1

    scores = defaultdict(float)
    matched = []

    for rule_id, count in candidate_counts.items():
        rule = rules_store[rule_id]
        if count == rule["ant_len"]:
            score = rule["confidence"] * (1.0 + 0.15 * math.log(1 + rule["ant_len"]))
            scores[rule["label"]] += score

            matched.append((rule["label"], rule["confidence"], rule["support"], rule["ant_len"], rule["antecedent"]))

    matched.sort(key=lambda x: x[1], reverse=True)
    matched = matched[:topn_match]

    if not scores:
        return None, [], {}

    pred = max(scores.items(), key=lambda x: x[1])[0]
    return pred, matched, dict(scores)


def train_associative_classifier(
    root_dir,
    k_keywords=30,
    min_df=2,
    max_df=0.9,
    ngram_range=(1, 2),
    min_support=0.02,
    max_len=4,
    min_conf=0.6,
    stopwords_vi=None
):
    df = load_dataset_from_folders(root_dir)
    if df.empty:
        raise ValueError("Không tìm thấy dữ liệu. Hãy kiểm tra cấu trúc thư mục và file .txt")

    keywords_list, vectorizer = top_k_keywords_per_doc_tfidf(
        df["text"].tolist(),
        k=k_keywords,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        stopwords_vi=stopwords_vi
    )

    transactions = build_transactions(keywords_list, df["label"].tolist())

    fi, items = mine_frequent_itemsets(transactions, min_support=min_support, max_len=max_len)
    labels_set = set(df["label"].unique().tolist())

    rules = generate_label_rules(fi, items, labels_set=labels_set, min_conf=min_conf)

    model = {
        "df": df,
        "vectorizer": vectorizer,
        "rules": rules,
        "labels_set": labels_set,
        "params": {
            "k_keywords": k_keywords,
            "min_df": min_df,
            "max_df": max_df,
            "ngram_range": ngram_range,
            "min_support": min_support,
            "max_len": max_len,
            "min_conf": min_conf,
        }
    }
    return model


def extract_keywords_for_new_doc(text: str, vectorizer: TfidfVectorizer, k=30):
    text = normalize_text_vi(text)
    X = vectorizer.transform([text])
    vocab = np.array(vectorizer.get_feature_names_out())

    row = X.getrow(0)
    if row.nnz == 0:
        return []

    idx = row.indices
    vals = row.data
    top_idx = idx[np.argsort(-vals)[:k]]
    kws = vocab[top_idx].tolist()
    return [kw.replace(" ", "_") for kw in kws]


def save_model_to_disk(model, filename_pkl="model_luat.pkl", filename_csv="danh_sach_luat.csv"):
    with open(filename_pkl, "wb") as f:
        pickle.dump(model, f)
    print(f">>> Đã lưu trọn bộ model vào file: {filename_pkl}")
    model["rules"].to_csv(filename_csv, index=False, encoding="utf-8-sig")
    print(f">>> Đã xuất danh sách luật ra file: {filename_csv}")


def load_model_from_disk(filename_pkl="model_luat.pkl"):
    try:
        with open(filename_pkl, "rb") as f:
            model = pickle.load(f)
        print(">>> Đã load model thành công!")
        return model
    except FileNotFoundError:
        print(">>> Không tìm thấy file model. Vui lòng train lại!")
        return None

def train_and_evaluate(
    train_root="Train_full",
    test_root="Test_full",
    model_path="model_luat.pkl"
):
    print("=== BẮT ĐẦU HUẤN LUYỆN ===")
    model = train_associative_classifier(
        train_root,
        k_keywords=25,
        min_conf=0.6
    )

    print("\n=== TỐI ƯU HÓA ĐỂ ĐÁNH GIÁ TEST ===")
    print("Đang đánh chỉ mục luật (Indexing Rules)...")
    rule_index, rules_store = build_rule_index(model["rules"])

    print("\n=== ĐÁNH GIÁ TRÊN TẬP TEST (Batch Mode) ===")
    test_df = load_dataset_from_folders(test_root)

    print("Đang trích xuất từ khóa cho toàn bộ tập test...")
    texts_clean = [normalize_text_vi(t) for t in tqdm(test_df["text"], desc="Normalize")]

    vectorizer = model["vectorizer"]
    X_test = vectorizer.transform(texts_clean)
    vocab = np.array(vectorizer.get_feature_names_out())
    k_val = model["params"]["k_keywords"]

    y_true, y_pred = [], []

    for i in tqdm(range(X_test.shape[0]), desc="Dự đoán nhanh"):
        row_vec = X_test.getrow(i)

        if row_vec.nnz == 0:
            kws = []
        else:
            idx = row_vec.indices
            vals = row_vec.data
            top_idx = idx[np.argsort(-vals)[:k_val]]
            kws_raw = vocab[top_idx].tolist()
            kws = [kw.replace(" ", "_") for kw in kws_raw]

        pred, _, _ = predict_label_fast(kws, rule_index, rules_store)

        if pred is None:
            pred = "UNKNOWN"

        y_true.append(test_df.iloc[i]["label"])
        y_pred.append(pred)

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred))

    save_model_to_disk(model, filename_pkl=model_path)
    print("=== HOÀN TẤT ===")

    return model

def predict_new_text(text, model_path="model_luat.pkl"):
    model = load_model_from_disk(model_path)
    if model is None:
        print("Không thể load model. Hãy train trước.")
        return None

    vectorizer = model["vectorizer"]
    rules_df = model["rules"]
    k_val = model["params"]["k_keywords"]

    rule_index, rules_store = build_rule_index(rules_df)

    kws = extract_keywords_for_new_doc(text, vectorizer, k=k_val)

    pred, matched_rules, _ = predict_label_fast(kws, rule_index, rules_store)

    return pred

train_and_evaluate()

text_test = "Cầu thủ ghi bàn thắng quyết định vào lưới đối phương"
label = predict_new_text(text_test)
print("Kết quả dự đoán:", label)