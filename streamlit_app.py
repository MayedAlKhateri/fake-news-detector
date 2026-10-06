import os
import re
import joblib
import numpy as np
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification


st.set_page_config(
    page_title="Fake News Detector",
    layout="centered",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_MODEL_FOLDER = os.path.join(BASE_DIR, "fake_news_hybrid")
HF_MODEL_REPO = "Mayeds/fake-news-detector-model"
MAX_LENGTH = 256


@st.cache_resource
def load_models():
    if os.path.exists(os.path.join(LOCAL_MODEL_FOLDER, "bert_model")):
        model_folder = LOCAL_MODEL_FOLDER
    else:
        from huggingface_hub import snapshot_download
        model_folder = snapshot_download(HF_MODEL_REPO)

    bert_folder = os.path.join(model_folder, "bert_model")
    tokenizer = AutoTokenizer.from_pretrained(bert_folder, local_files_only=True)
    bert_model = AutoModelForSequenceClassification.from_pretrained(bert_folder, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model.to(device)
    bert_model.eval()

    tfidf_vectorizer = joblib.load(os.path.join(model_folder, "tfidf_vectorizer.joblib"))
    lr_model = joblib.load(os.path.join(model_folder, "logistic_regression.joblib"))
    config = joblib.load(os.path.join(model_folder, "ensemble_config.joblib"))
    return tokenizer, bert_model, tfidf_vectorizer, lr_model, config, device


(
    tokenizer,
    bert_model,
    tfidf_vectorizer,
    lr_model,
    config,
    device,
) = load_models()

BERT_WEIGHT = config.get("bert_weight", 0.5)
LR_WEIGHT = config.get("lr_weight", 0.5)


def normalize_text(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def safe_probability(prob_vector, class_index=0):
    arr = np.atleast_1d(np.asarray(prob_vector, dtype=float))
    if arr.size == 0:
        return 0.0
    idx = int(class_index)
    if idx >= arr.size:
        idx = arr.size - 1
    return float(arr[idx])


def get_label_for_index(index):
    index = int(index)
    label_map = getattr(bert_model.config, "id2label", {})
    label = label_map.get(str(index), label_map.get(index))
    if label:
        return str(label).upper()
    return "FAKE" if index == 0 else "REAL"


def get_model_scores(title, article):
    article_text = normalize_text(article)
    title_text = normalize_text(title)
    combined_text = " ".join([title_text, article_text]).strip()

    inputs = tokenizer(
        title_text,
        article_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = bert_model(**inputs)
        bert_probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    tfidf_features = tfidf_vectorizer.transform([combined_text])
    lr_probs = lr_model.predict_proba(tfidf_features)[0]

    final_probs = (BERT_WEIGHT * bert_probs) + (LR_WEIGHT * lr_probs)
    final_probs = final_probs / np.sum(final_probs)

    prediction_index = int(np.argmax(final_probs))
    confidence = float(final_probs[prediction_index]) * 100
    result = get_label_for_index(prediction_index)

    return {
        "title": title_text,
        "article": article_text,
        "combined_text": combined_text,
        "bert_probs": bert_probs,
        "lr_probs": lr_probs,
        "final_probs": final_probs,
        "prediction_index": prediction_index,
        "result": result,
        "confidence": confidence,
    }


def extract_top_terms(text, limit=8):
    words = re.findall(r"[A-Za-z][A-Za-z'-]{2,}", text.lower())
    stop_words = {
        "the", "this", "that", "with", "from", "into", "your", "about", "after",
        "have", "been", "more", "their", "there", "would", "could", "should",
        "through", "while", "over", "under", "than", "them", "they", "themself",
        "what", "when", "where", "which", "who", "why", "how", "just", "also",
        "very", "much", "only", "made", "like", "said", "were", "was", "will",
        "news", "story", "article", "headline"
    }
    filtered = [word for word in words if word not in stop_words and len(word) > 2]

    from collections import Counter
    single_counter = Counter(filtered)
    phrases = []
    for i in range(len(filtered) - 1):
        pair = f"{filtered[i]} {filtered[i + 1]}"
        if len(pair.split()) == 2 and filtered[i] != filtered[i + 1]:
            phrases.append(pair)
    phrase_counter = Counter(phrases)

    ranked_phrases = [phrase for phrase, _ in phrase_counter.most_common()]
    ranked_words = [word for word, _ in single_counter.most_common()]

    final_terms = []
    for item in ranked_phrases + ranked_words:
        if not item:
            continue

        item_tokens = set(str(item).split())
        if not item_tokens:
            continue

        reject = False
        for existing in list(final_terms):
            existing_tokens = set(str(existing).split())
            overlap = item_tokens & existing_tokens
            if not overlap:
                continue

            if item == existing:
                reject = True
                break

            shorter = min(len(item_tokens), len(existing_tokens))
            overlap_ratio = len(overlap) / shorter if shorter else 0
            if overlap_ratio >= 0.5:
                reject = True
                break

            if len(item_tokens) > len(existing_tokens) and len(overlap) >= max(1, len(existing_tokens)):
                final_terms.remove(existing)
                break

        if reject:
            continue

        final_terms.append(item)
        if len(final_terms) >= limit:
            break

    if not final_terms:
        final_terms = filtered[:limit]

    return final_terms


def get_reason_summary(result, title, article, top_terms):
    combined = f"{title} {article}".lower()
    suspicious_tokens = [
        "urgent", "breaking", "shocking", "miracle", "secret", "viral", "click", "exposed",
        "must", "share", "warning", "scam", "conspiracy", "unbelievable", "hoax", "banned"
    ]
    trusted_tokens = [
        "according", "report", "official", "study", "data", "analysis", "statement",
        "confirmed", "evidence", "research", "agency", "document", "court", "police", "investigation"
    ]
    suspicious_hits = [term for term in suspicious_tokens if term in combined]
    trusted_hits = [term for term in trusted_tokens if term in combined]

    if result == "FAKE":
        hits = suspicious_hits if suspicious_hits else (top_terms[:3] if top_terms else [])
        if hits:
            return (
                "The model is leaning toward fake content because the article includes attention-grabbing, alarmist wording and patterns often used in misinformation campaigns. "
                f"The strongest cues include: {', '.join(hits[:4])}."
            )
        return (
            "The model is flagging this as fake because the wording is highly emotional and lacks the grounding cues typically seen in reliable reporting."
        )

    if trusted_hits:
        return (
            "The model is leaning real overall because the article includes grounded source language, specific references, and evidence-like phrasing. "
            f"It is still worth noting that a few attention-grabbing phrases are present, such as: {', '.join((suspicious_hits[:3] if suspicious_hits else top_terms[:3])[:3])}."
        )
    return (
        "The model is leaning real because the article reads with more factual structure, specificity, and neutral tone than typical misinformation patterns."
    )


def tooltip_label(label, tooltip):
    return (
        f'<span style="display:inline-flex; align-items:center; gap:6px; color:#dfeafc; '
        f'font-size:11px; font-weight:700; letter-spacing:0.08em; text-transform:uppercase;">'
        f'{label}<span title="{tooltip}" style="display:inline-flex; align-items:center; justify-content:center; '
        f'width:14px; height:14px; border-radius:50%; background: rgba(138, 169, 255, 0.18); '
        f'border:1px solid rgba(138,169,255,0.38); color:#dfeafc; font-size:10px; font-weight:800; cursor:pointer;">i</span></span>'
    )


def get_top_lr_features(result, feature_names, class_index):
    if not hasattr(lr_model, "coef_") or not hasattr(tfidf_vectorizer, "get_feature_names_out"):
        return []

    coef = np.asarray(lr_model.coef_)
    if coef.size == 0:
        return []

    if coef.ndim == 1:
        weights = coef
    else:
        safe_index = int(class_index)
        safe_index = max(0, min(safe_index, coef.shape[0] - 1))
        weights = coef[safe_index]

    feature_names = np.asarray(feature_names)
    ranked = np.argsort(np.abs(weights))[::-1]
    selected = []
    seen = set()
    for idx in ranked:
        name = str(feature_names[idx])
        if name and len(name) > 2 and name not in seen:
            selected.append({"word": name, "weight": float(weights[idx])})
            seen.add(name)
        if len(selected) >= 6:
            break
    return selected


def render_probability_bar(label, value, positive_color):
    pct = max(0.0, min(100.0, float(value)))
    return f"""
    <div style="margin-bottom: 12px;">
        <div style="display:flex; justify-content:space-between; margin-bottom:6px; font-size:13px; color:#dfe7f5;">
            <span>{label}</span>
            <strong>{pct:.1f}%</strong>
        </div>
        <div style="height:8px; border-radius:999px; background: rgba(255,255,255,0.08); overflow:hidden;">
            <div style="width:{pct:.1f}%; height:100%; border-radius:999px; background: {positive_color};"></div>
        </div>
    </div>
    """


def build_cue_chips(top_terms, lr_signals, article_text):
    signal_map = {}
    for entry in lr_signals or []:
        word = str(entry.get("word", "")).strip().lower()
        if word:
            signal_map[word] = abs(float(entry.get("weight", 0.0)))

    max_signal = max(signal_map.values(), default=1.0)
    text_lower = (article_text or "").lower()
    suspicious_tokens = [
        "urgent", "breaking", "shocking", "miracle", "secret", "viral", "click", "exposed",
        "must", "share", "warning", "scam", "conspiracy", "unbelievable", "hoax", "banned"
    ]
    trusted_tokens = [
        "according", "report", "official", "study", "data", "analysis", "statement",
        "confirmed", "evidence", "research", "agency", "document", "court", "police", "investigation"
    ]

    chips = []

    for term in top_terms or []:
        text = str(term).strip()
        if not text:
            continue
        phrase_norm = text.lower()
        freq = len(re.findall(rf"\b{re.escape(phrase_norm)}\b", text_lower))
        if freq == 0:
            freq = text_lower.count(phrase_norm)
        if freq == 0:
            freq = 1

        weight_score = signal_map.get(phrase_norm, 0.0) / max_signal if max_signal else 0.0
        phrase_word_count = max(1, len(phrase_norm.split()))
        suspicious_bonus = sum(8 for token in suspicious_tokens if token in phrase_norm)
        trusted_bonus = sum(6 for token in trusted_tokens if token in phrase_norm)
        length_bonus = min(28, phrase_word_count * 12 + len(text) // 3)

        score = 18 + (freq * 10) + int(weight_score * 45) + length_bonus + suspicious_bonus - trusted_bonus
        score = max(22, min(96, score))
        chips.append({"text": text, "score": int(score)})

    if not chips:
        return []

    chips = sorted(chips, key=lambda x: x["score"], reverse=True)
    return chips[:6]


st.markdown(
    """
    <style>
        .stApp {
            background: radial-gradient(circle at top, rgba(34, 78, 138, 0.42) 0%, rgba(13, 24, 39, 0.95) 28%, #030a13 100%);
            color: #edf4ff;
        }
        .block-container {
            max-width: 960px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        #MainMenu, footer, [data-testid="stDecoration"] { visibility: hidden; }
        div[data-testid="stTextInput"] input, div[data-testid="stTextArea"] textarea {
            background: rgba(11, 20, 32, 0.9);
            color: #edf4ff;
            border: 1px solid rgba(132, 162, 214, 0.35);
            border-radius: 14px;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,0.02);
            padding: 0.9rem 1rem;
        }
        div[data-testid="stTextInput"] input:focus, div[data-testid="stTextArea"] textarea:focus {
            border-color: rgba(118, 146, 255, 0.9);
            box-shadow: 0 0 0 1px rgba(125, 146, 255, 0.25);
        }
        div[data-testid="stTextInput"] input::placeholder, div[data-testid="stTextArea"] textarea::placeholder {
            color: #7d8aa3;
        }
        div.stButton > button {
            background: linear-gradient(90deg, #6e8cff 0%, #8a6df3 100%);
            color: white;
            border: none;
            border-radius: 14px;
            padding: 0.8rem 1.2rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            box-shadow: 0 12px 30px rgba(84, 112, 255, 0.35);
            transition: all 0.2s ease;
        }
        div.stButton > button:hover {
            box-shadow: 0 16px 36px rgba(84, 112, 255, 0.45);
            filter: brightness(1.04);
        }
        .metric-shell {
            background: rgba(15, 24, 36, 0.72);
            border: 1px solid rgba(137, 166, 214, 0.18);
            border-radius: 18px;
            padding: 1rem 1rem 0.9rem;
            margin-top: 1.2rem;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.8rem;
        }
        .metric-panel {
            background: rgba(255,255,255,0.015);
            border: 1px solid rgba(155, 178, 216, 0.10);
            border-radius: 12px;
            padding: 0.8rem 0.75rem 0.7rem;
            min-height: 100px;
        }
        .metric-panel .metric-label {
            display: flex;
            align-items: center;
            gap: 6px;
            margin-bottom: 10px;
            color: #8ea3c0;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .metric-panel .metric-value {
            font-size: 2rem;
            font-weight: 800;
            color: #edf4ff;
            line-height: 1.1;
        }
        .result-card {
            border-radius: 18px;
            padding: 1.1rem 1rem;
            margin-top: 1.5rem;
            border: 1px solid rgba(128, 159, 214, 0.18);
            box-shadow: 0 14px 30px rgba(7, 15, 25, 0.28);
        }
        .result-card.real {
            background: linear-gradient(135deg, rgba(8, 35, 32, 0.9), rgba(10, 29, 38, 0.94));
            border-color: rgba(72, 220, 182, 0.45);
        }
        .result-card.fake {
            background: linear-gradient(135deg, rgba(39, 17, 25, 0.94), rgba(18, 15, 29, 0.92));
            border-color: rgba(255, 136, 176, 0.46);
        }
        .tag {
            display:inline-block;
            padding: 7px 14px;
            border-radius:999px;
            font-size: 12px;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .tag.real {
            background: rgba(48, 168, 128, 0.2);
            color: #8ff4d5;
            border: 1px solid rgba(72, 220, 182, 0.35);
        }
        .tag.fake {
            background: rgba(183, 67, 104, 0.18);
            color: #ffb5cf;
            border: 1px solid rgba(255, 139, 178, 0.35);
        }
        .chip {
            display:inline-flex;
            align-items:center;
            gap:8px;
            background: rgba(190, 208, 234, 0.07);
            color:#ebf4ff;
            border:1px solid rgba(152, 178, 215, 0.18);
            border-radius:999px;
            padding: 6px 10px 6px 12px;
            margin: 0 8px 8px 0;
            font-size: 12px;
            font-weight: 600;
            min-height: 32px;
            transition: all 0.2s ease;
        }
        .chip.strong {
            background: linear-gradient(135deg, rgba(255, 117, 153, 0.18), rgba(255, 141, 113, 0.10));
            border-color: rgba(255, 142, 169, 0.45);
            box-shadow: 0 0 0 1px rgba(255, 142, 169, 0.10), 0 10px 18px rgba(255, 116, 152, 0.10);
        }
        .chip-text {
            display:inline-block;
            line-height:1;
        }
        .chip-score {
            display:inline-flex;
            align-items:center;
            justify-content:center;
            min-width: 56px;
            height: 20px;
            padding: 0 7px;
            border-radius: 999px;
            background: rgba(110, 140, 255, 0.16);
            color: #dfeaff;
            font-size: 9px;
            font-weight: 800;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        .chip.strong .chip-score {
            background: rgba(255, 128, 166, 0.18);
            color: #ffd9e7;
        }
        .info-box {
            background: rgba(15, 25, 38, 0.88);
            border: 1px solid rgba(134, 166, 216, 0.2);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            margin-top: 1rem;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
        }
        .section-header {
            display:flex;
            align-items:center;
            gap:8px;
            margin-bottom: 0.8rem;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.08em;
            color: #9bb5d8;
            text-transform: uppercase;
        }
        .chip-row {
            display:flex;
            flex-wrap:wrap;
            gap: 0;
            margin-top: 8px;
        }
        .info-box strong {
            color: #ffffff;
        }
        h3, h4 {
            color: #eef5ff;
        }
        [data-testid="stMetricValue"] {
            color: #ebf4ff;
            font-size: 1.8rem;
            font-weight: 700;
        }
        [data-testid="stMetricLabel"] {
            color: #8ea3c0;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-size: 0.7rem;
        }
        .stAlert {
            background: rgba(10, 17, 27, 0.8);
            border: 1px solid rgba(135, 166, 215, 0.2);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

header_html = """
<div style="display:flex; align-items:center; gap:14px; margin: 0 0 1.5rem 0;">
    <div style="width:46px; height:46px; border-radius:12px; background: linear-gradient(135deg,#4f78ff,#915cf6); display:flex; align-items:center; justify-content:center; color:#fff; font-size:22px;">✦</div>
    <div>
        <div style="color:#eef4ff; font-size: 1.25rem; font-weight: 700;">Fake News Detector</div>
        <div style="color:#8c9bb7; font-size: 0.78rem; margin-top: 4px;">AI-powered news credibility prediction</div>
    </div>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)


title = st.text_input("Article Title (optional)", placeholder="Enter the news headline")
article = st.text_area("Article Text", placeholder="Paste the full news article here...")

analyze = st.button("Analyze Article", use_container_width=True)

if analyze:
    if not article.strip():
        st.warning("Please enter the article text.")
    else:
        with st.spinner("Analyzing article..."):
            result_data = get_model_scores(title, article)

        result = result_data["result"]
        confidence = float(result_data["confidence"])
        final_probs = result_data["final_probs"]
        bert_probs = result_data["bert_probs"]
        lr_probs = result_data["lr_probs"]
        final_probs_1d = np.atleast_1d(np.asarray(final_probs, dtype=float))
        final_fake_score = float(safe_probability(final_probs_1d, 0) * 100)
        final_real_score = float(safe_probability(final_probs_1d, 1 if final_probs_1d.size > 1 else 0) * 100)

        top_terms = extract_top_terms(result_data["combined_text"]) or ["source", "report", "claim"]
        summary = get_reason_summary(result, result_data["title"], result_data["article"], top_terms)

        feature_names = tfidf_vectorizer.get_feature_names_out()
        coef = np.asarray(lr_model.coef_)
        if coef.ndim == 1:
            class_index = 0
        else:
            class_index = 0 if result == "FAKE" else 1
            class_index = min(class_index, max(coef.shape[0] - 1, 0))
        lr_signals = get_top_lr_features(result, feature_names, class_index)
        if not lr_signals:
            lr_signals = [{"word": term, "weight": 1.0} for term in top_terms[:5]]

        tag_class = "real" if result == "REAL" else "fake"
        result_label = "REAL NEWS" if result == "REAL" else "FAKE NEWS"
        result_color = "#7ef1d8" if result == "REAL" else "#ff9ab5"

        st.markdown(
            f"""
            <div class="result-card {tag_class}">
                <div style="display:flex; justify-content:space-between; align-items:center; gap: 12px; flex-wrap:wrap;">
                    <span class="tag {tag_class}">{result_label}</span>
                    <span style="color:#eaf2ff; font-size:14px;">Confidence: <strong>{confidence:.2f}%</strong></span>
                </div>
                <div style="margin-top: 22px; background: rgba(255,255,255,0.06); height: 8px; border-radius: 999px; overflow:hidden;">
                    <div style="width:{confidence:.2f}%; height:100%; background:{result_color}; border-radius:999px;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="metric-shell">
                <div class="metric-panel">
                    <div class="metric-label">{tooltip_label('BERT', 'The transformer model score: how strongly the language model sees the article as real or fake based on wording and context.')}</div>
                    <div class="metric-value">{safe_probability(bert_probs, 1 if result == 'REAL' else 0) * 100:.1f}%</div>
                </div>
                <div class="metric-panel">
                    <div class="metric-label">{tooltip_label('LogReg', 'The logistic regression score: how the keyword-based model weights article terms like “launch”, “official”, or “warning”.')}</div>
                    <div class="metric-value">{safe_probability(lr_probs, 1 if result == 'REAL' else 0) * 100:.1f}%</div>
                </div>
                <div class="metric-panel">
                    <div class="metric-label">{tooltip_label('Final', 'The combined ensemble score after blending the BERT and logistic-regression predictions into one final verdict.')}</div>
                    <div class="metric-value">{confidence:.1f}%</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="info-box">
                <div class="section-header">
                    {tooltip_label('Model reasoning', 'This explains the overall decision in plain language by combining the strongest signals from the article and the ensemble model.')}
                </div>
                <div style="color:#eaf3ff; line-height:1.6; font-size: 0.98rem;">{summary}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class='info-box'>
                <div class='section-header'>
                    {tooltip_label('Signal breakdown', 'The probability bars show how strongly the combined model favors fake versus real after both models are blended.')}
                </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(render_probability_bar("Fake likelihood", final_fake_score if result == "FAKE" else 100 - final_real_score, "#ff7ca8"), unsafe_allow_html=True)
        st.markdown(render_probability_bar("Real likelihood", final_real_score if result == "REAL" else 100 - final_fake_score, "#63ecce"), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        cue_items = build_cue_chips(top_terms, lr_signals, result_data["combined_text"])
        best_cue_score = max((item["score"] for item in cue_items), default=0)
        cue_html = []
        for item in cue_items:
            css_class = "chip strong" if item["score"] == best_cue_score else "chip"
            cue_html.append(f"<span class='{css_class}'><span class='chip-text'>{item['text']}</span><span class='chip-score'>Influence {item['score']}</span></span>")

        st.markdown(
            f"""
            <div class='info-box'>
                <div class='section-header'>
                    {tooltip_label('Top cues in the article', 'These are the strongest article signals. The influence score reflects how strongly that cue contributed to the model decision, not the final prediction probability.')}
                </div>
                <div class='chip-row'>
                    {''.join(cue_html)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if lr_signals:
            weighted_items = []
            max_w = max(abs(float(entry.get('weight', 0.0))) for entry in lr_signals) if lr_signals else 1.0
            for entry in lr_signals[:6]:
                word = str(entry.get('word', '')).strip()
                if not word:
                    continue
                weight = abs(float(entry.get('weight', 0.0)))
                score = max(20, min(95, int((weight / max_w) * 100)))
                weighted_items.append({"text": word, "score": score})
            if weighted_items:
                max_weight_score = max(item["score"] for item in weighted_items)
                weighted_html = []
                for item in weighted_items:
                    css_class = "chip strong" if item["score"] == max_weight_score else "chip"
                    weighted_html.append(f"<span class='{css_class}'><span class='chip-text'>{item['text']}</span><span class='chip-score'>Influence {item['score']}</span></span>")
                st.markdown(
                    f"""
                    <div class='info-box' style='margin-top: 1rem;'>
                        <div class='section-header'>
                            {tooltip_label('Model-weighted signal words', 'These are the words the logistic-regression model weighted most heavily for the chosen class. Higher influence means the model relied on that cue more heavily.')}
                        </div>
                        <div class='chip-row'>
                            {''.join(weighted_html)}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.caption("The model uses the saved class labels from the trained ensemble, so the output stays aligned with the real model rather than a manual guess.")
