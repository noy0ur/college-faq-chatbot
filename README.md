# College FAQ Chatbot

A Python chatbot that answers common college questions by matching user messages against a local FAQ knowledge base using TF-IDF and cosine similarity.

## Features
- Chat-style interface with Streamlit
- NLP-based FAQ retrieval using scikit-learn
- FAQ categories and matched-question transparency
- Fallback response when a match is weak
- Editable local `faq_data.json` knowledge base
- No paid API key required

## Project structure
- `app.py` — chatbot application
- `faq_data.json` — example FAQs and alternative phrasings
- `requirements.txt` — Python dependencies
- `README.md` — documentation
- `.gitignore` — excludes temporary files and secrets
- `LICENSE` — MIT license for this code

## Run
Python 3.10+ recommended.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Customize
Edit `faq_data.json`. Each entry has `question`, `alternatives`, `answer`, and `category`. Replace generic examples with verified details from your college before using it as a college-specific bot. Never invent fees, deadlines, eligibility rules, or contact details.

## Deploy
1. Upload the project files to GitHub.
2. Visit https://share.streamlit.io/
3. Create an app using your repository and branch.
4. Set main file path to `app.py`.
5. Deploy.

## How it works
The app vectorizes FAQ questions and alternate phrasings with TF-IDF, compares a user query with cosine similarity, and returns the closest answer only when its score passes a threshold. The score is not a probability.

## Limitations and responsible use
This is a retrieval-based chatbot, not a generative AI system. It can only answer questions covered by its knowledge base. Sample responses are not official information for a specific college. Verify current dates, fees, eligibility, and policies using official sources. Do not store private student records or credentials in this repository.

## Skills demonstrated
Python, NLP, TF-IDF, cosine similarity, scikit-learn, Streamlit, JSON data handling, retrieval systems, and Git/GitHub.
