import json
from pathlib import Path
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).parent
FAQ_PATH = BASE_DIR / "faq_data.json"
CONFIDENCE_THRESHOLD = 0.18

st.set_page_config(page_title="College FAQ Chatbot", page_icon="🎓", layout="centered")

@st.cache_data
def load_faqs():
    with FAQ_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [x for x in data if all(k in x for k in ("question", "answer", "category"))]

def build_index(faqs):
    phrases, positions = [], []
    for i, faq in enumerate(faqs):
        for phrase in [faq["question"]] + faq.get("alternatives", []):
            if str(phrase).strip():
                phrases.append(str(phrase).strip())
                positions.append(i)
    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(phrases)
    return vectorizer, matrix, positions

def get_answer(question, faqs, vectorizer, matrix, positions):
    if not question.strip():
        return None, 0.0
    scores = cosine_similarity(vectorizer.transform([question.strip()]), matrix).ravel()
    best = int(scores.argmax())
    score = float(scores[best])
    if score < CONFIDENCE_THRESHOLD:
        return None, score
    return faqs[positions[best]], score

st.title("🎓 College FAQ Chatbot")
st.write("Ask questions about admissions, courses, fees, scholarships, documents, and campus facilities.")
st.caption("Demo using a local FAQ knowledge base and TF-IDF similarity. Answers depend on the FAQ content.")

try:
    faqs = load_faqs()
    if not faqs:
        st.error("No valid FAQs found in faq_data.json.")
        st.stop()
    vectorizer, matrix, positions = build_index(faqs)
except FileNotFoundError:
    st.error("faq_data.json is missing. Add it beside app.py.")
    st.stop()
except Exception as exc:
    st.error(f"Could not load FAQs: {exc}")
    st.stop()

with st.sidebar:
    st.header("About")
    st.write(f"Loaded **{len(faqs)} FAQs**.")
    st.write("Uses TF-IDF and cosine similarity; no paid API key is required.")
    st.subheader("Try asking")
    for sample in ["How do I apply for admission?", "Which documents are required?", "Are scholarships available?", "Where is the library?"]:
        st.write("• " + sample)
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("category"):
            st.caption("FAQ category: " + msg["category"])

prompt = st.chat_input("Type your college question...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    answer, score = get_answer(prompt, faqs, vectorizer, matrix, positions)
    if answer is None:
        response = ("I couldn't find a sufficiently close answer in my FAQ knowledge base. "
                    "Try different wording or contact the relevant college office. "
                    "For accurate details, check your college's official website.")
        category = None
    else:
        response, category = answer["answer"], answer["category"]
    with st.chat_message("assistant"):
        st.markdown(response)
        if category:
            st.caption("FAQ category: " + category)
            with st.expander("How was this answer selected?"):
                st.write("Matched FAQ: " + answer["question"])
                st.write(f"Similarity score: {score:.3f}")
                st.caption("This score is not a probability or a guarantee of correctness.")
    st.session_state.messages.append({"role": "assistant", "content": response, "category": category})
