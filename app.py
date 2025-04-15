from flask import Flask, request, jsonify, render_template
import re
import heapq
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

app = Flask(__name__)

# Download necessary NLTK data (will only download once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def analyze_toxicity(text):
    # Normalize the text for analysis
    normalized_text = text.lower().strip()
    
    # Define categories of problematic patterns - expanded for better detection
    toxic_patterns = {
        "hate": [
            r"\b(hate|despise|detest)\b",
            r"\b(racist|racism|bigot|bigotry)\b", 
            r"\b(sexist|sexism|misogyn|misandr)\b",
            r"\b(homophob|transphob)\b",
            r"\b(xenophob)\b"
        ],
        "insults": [
            r"\b(stupid|idiot|dumb|moron)\b", 
            r"\b(loser|pathetic|worthless|useless)\b",
            r"\b(ugly|fat|disgusting)\b",
            r"\b(shut up|stfu)\b",
            r"\b(ass|asshole)\b",
            r"\b(jerk|douche|dick)\b"
        ],
        "profanity": [
            r"\b(damn|hell|crap)\b",
            r"\b(shit|fuck|bitch)\b",
            r"\b(wtf|stfu|fu|fu?k|sh\*t|b\*tch)\b",
            r"\b(f\*\*k|s\*\*t)\b"
        ],
        "threats": [
            r"\b(kill|hurt|harm|punch|attack|beat)\b", 
            r"\b(die|death|dead|suicide)\b",
            r"\b(threat|threaten)\b",
            r"\b(destroy|ruin|end you)\b"
        ],
        "identity_attacks": [
            r"\b(retard|retarded)\b",
            r"\b(n[i!1]gg[ae]r|f[a@]gg?[o0]t)\b",
            r"\b(ch[i!1]nk|sp[i!1]c|k[i!1]ke|towelhead)\b",
            r"\b(wh[o0]re|sl[u]t|c[u]nt)\b"
        ],
        "harassment": [
            r"\b(stalk|harass|bully)\b",
            r"\b(creep|creepy)\b",
            r"\b(troll|trolling)\b"
        ],
        "obscene": [
            r"\b(porn|pornography)\b",
            r"\b(sex|sexual)\b",
            r"\b(penis|vagina|dick)\b"
        ],
        # New category specifically for offensive phrases
        "offensive_phrases": [
            # Common offensive phrases - using looser boundaries
            r"(fuck off|fuck you|fuck it|fuck that)",
            r"(jerk off|jack off|get off|piss off)",
            r"(go to hell|go fuck yourself|go die)",
            r"(shut up|shut the fuck up|shut it)",
            r"(screw you|screw off|screw that)",
            r"(suck my|suck it|suck a)",
            r"(kiss my ass|my ass|up yours)",
            r"(eat shit|eat my)"
        ]
    }

    # Track detected patterns and toxic words
    detected_patterns = []
    toxic_words = []
    
    # Check each category
    total_weight = 0
    category_weights = {
        "hate": 0.8,
        "insults": 0.6,
        "profanity": 0.5,
        "threats": 0.9,
        "identity_attacks": 0.9,
        "harassment": 0.7,
        "obscene": 0.6,
        "offensive_phrases": 0.8  # High weight for offensive phrases
    }

    # First, check for offensive phrases (do this first to catch multi-word expressions)
    phrase_patterns = toxic_patterns["offensive_phrases"]
    for pattern in phrase_patterns:
        matches = re.finditer(pattern, normalized_text, re.IGNORECASE)
        for match in matches:
            detected_patterns.append("offensive_phrases")
            total_weight += category_weights["offensive_phrases"]
            toxic_words.append(match.group())

    # Then check each other category for matches
    for category, patterns in toxic_patterns.items():
        # Skip offensive_phrases as we already processed them
        if category == "offensive_phrases":
            continue
            
        category_weight = category_weights[category]
        
        for pattern in patterns:
            matches = re.finditer(pattern, normalized_text, re.IGNORECASE)
            for match in matches:
                detected_patterns.append(category)
                total_weight += category_weight
                toxic_words.append(match.group())

    # Calculate final score (normalized between 0 and 1)
    # Lower the divisor to make the score more sensitive
    max_possible_weight = 3.0  # This is lower than the actual sum of weights to increase sensitivity
    score = min(total_weight / max_possible_weight, 1)

    return {
        "score": score,
        "detected_patterns": list(set(detected_patterns)),  # Remove duplicates
        "toxic_words": list(set(toxic_words))  # Remove duplicates
    }

def censor_toxic_words(text, toxic_words):
    """Censor toxic words by replacing characters with asterisks (*) except the first and last character"""
    censored_text = text
    for word in toxic_words:
        # Skip words that are too short
        if len(word) <= 2:
            continue
            
        # Create censored version (keep first letter, replace middle with *, keep last letter)
        if len(word) <= 4:
            # For short words, just keep first letter
            censored = word[0] + '*' * (len(word) - 1)
        else:
            censored = word[0] + '*' * (len(word) - 2) + word[-1]
            
        # Replace all occurrences, preserving case
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        
        def replace_match(match):
            matched = match.group(0)
            if matched[0].isupper():
                return censored[0].upper() + censored[1:]
            return censored
            
        censored_text = pattern.sub(replace_match, censored_text)
    
    return censored_text

def summarize_text(text, num_sentences=3):
    """Generate a summary of the text using extractive summarization"""
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    # If text is too short, return it as is
    if len(sentences) <= num_sentences:
        return text
        
    # Tokenize words and calculate word frequencies
    stop_words = set(stopwords.words('english'))
    word_frequencies = {}
    
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word not in stop_words and word.isalnum():
                if word not in word_frequencies:
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1
    
    # Normalize word frequencies
    if word_frequencies:
        max_frequency = max(word_frequencies.values())
        for word in word_frequencies:
            word_frequencies[word] = word_frequencies[word] / max_frequency
    
    # Calculate sentence scores based on word frequencies
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        for word in word_tokenize(sentence.lower()):
            if word in word_frequencies:
                if i not in sentence_scores:
                    sentence_scores[i] = word_frequencies[word]
                else:
                    sentence_scores[i] += word_frequencies[word]
    
    # Get the top N sentences with highest scores
    summary_sentences_indices = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    summary_sentences_indices.sort()  # Sort to maintain original order
    
    # Combine the top sentences to form the summary
    summary = ' '.join([sentences[i] for i in summary_sentences_indices])
    
    return summary

def check_toxicity(text):
    try:
        # Normalize the text for analysis
        normalized_text = text.lower().strip()

        # Calculate toxicity score based on predefined patterns
        result = analyze_toxicity(normalized_text)
        score = result["score"]
        detected_patterns = result["detected_patterns"]
        toxic_words = result["toxic_words"]

        # Determine if the text is toxic based on the score threshold
        is_toxic = score > 0.3

        # Generate appropriate message
        message = "This comment may contain toxic content and could be harmful or offensive." if is_toxic else "This comment appears to be non-toxic and appropriate."

        # Add details about detected patterns if toxic
        if is_toxic and detected_patterns:
            message += " The system detected potentially problematic language patterns."
            
        # Censor toxic words in the original text
        censored_text = censor_toxic_words(text, toxic_words) if is_toxic else text
        
        # Generate summary if text is long enough (more than 3 sentences)
        summary = None
        if len(text.split()) > 50:  # Only summarize if more than 50 words
            try:
                summary = summarize_text(text)
                # Also censor the summary if needed
                if is_toxic:
                    summary = censor_toxic_words(summary, toxic_words)
            except Exception as e:
                print(f"Error generating summary: {e}")

        return {
            "is_toxic": is_toxic,
            "score": score,
            "message": message,
            "detected_patterns": detected_patterns if is_toxic else [],
            "censored_text": censored_text,
            "summary": summary
        }
    except Exception as e:
        print(f"Error analyzing text: {e}")
        raise Exception("Failed to analyze text for toxicity")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/check-toxicity', methods=['POST'])
def api_check_toxicity():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "Invalid request. Text is required."}), 400
        
    text = data['text']
    try:
        result = check_toxicity(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

print("Enhanced Toxic Comment Detector is running. Open http://127.0.0.1:5000 in your browser.")