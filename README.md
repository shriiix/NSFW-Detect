Enhanced Toxic Content Detector

A Python web application that detects toxic content in text, automatically censors offensive words, and generates text summaries using NLP techniques.

FEATURES
--------
- Toxic content detection across multiple categories
- Automatic censoring of offensive words (e.g., "stupid" → "s****d")
- Text summarization for longer content
- Modern, responsive UI with visual feedback
- REST API for integration with other applications

INSTALLATION
------------
# Clone repository (after creating on GitHub)
git clone https://github.com/yourusername/toxic-content-detector.git
cd toxic-content-detector

# Install dependencies
pip install flask nltk

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Run the application
python app.py

Then open http://127.0.0.1:5000 in your browser.

USAGE
-----
1. Enter text in the input area
2. Click "Analyze Content"
3. View toxicity score, categories, censored text, and summary

API USAGE
---------
import requests

response = requests.post('http://localhost:5000/api/check-toxicity', 
                        json={'text': 'Your text here'})
result = response.json()

TECHNICAL DETAILS
----------------
- Backend: Python with Flask
- NLP: NLTK for tokenization, stopword removal, and summarization
- Pattern Matching: Regular expressions for toxic content detection
- Frontend: HTML, CSS, JavaScript

LIBRARIES USED
-------------
- Flask: Web framework
- NLTK: Natural language processing
- re: Regular expressions for pattern matching
- heapq: Priority queue for summarization algorithm

EXAMPLE
-------
Input:
You're so stupid and worthless.

Output:
You're so s****d and w*******s.

LICENSE
-------
MIT License

Created with love by [Your Name]
