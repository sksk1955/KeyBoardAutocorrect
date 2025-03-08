from flask import Flask, request, jsonify, render_template
import pandas as pd
import textdistance
import re
from collections import Counter
import time

app = Flask(__name__)

# LOAD AND PROCSS THE VOCABLARY FRM THE OXFORD.TXT FILE
with open("Oxford.txt", encoding="utf8") as f:
    file_data = f.read().lower()
    words = re.findall(r"\w+", file_data)

# CREATE A SET OF UNIQUE WORDS AND A FREQUENCY DICTIONARY FOR THE WORDS
V = set(words)
word_freq_dict = Counter(words)
Total = sum(word_freq_dict.values())

# CALCULATE PROBABILITIES OF EACH WORD IN THE VOCABULARY
probs = {k: word_freq_dict[k] / Total for k in word_freq_dict.keys()}

# CACHE TO STORE RECENT AUTOCORRECTED WORDS WITH TIMESTAMP
cache = {}
CACHE_EXPIRY_TIME = 300  # CACHE EXPIRY TIME IN SECONDS (5 MINUTES)

# FUNCTION FOR JACCARD SIMILARITY-BASED AUTOCORRECT
def autocorrect_word(input_word, top_n=5):
    input_word = input_word.lower()

    # CHECK IF THE WORD IS IN THE CACHE AND IF IT IS STILL VALID
    if input_word in cache:
        cached_data, timestamp = cache[input_word]
        if time.time() - timestamp < CACHE_EXPIRY_TIME:
            # RETURN CACHED RESULT IF THE CACHE IS STILL VALID
            return cached_data
        else:
            # CACHE EXPIRED, REMOVE THE WORD FROM CACHE
            del cache[input_word]
    
    # IF THE WORD IS IN THE VOCABULARY, RETURN AN EMPTY LIST (NO SUGGESTIONS)
    if input_word in V:
        result = []
    else:
        # CALCULATE JACCARD SIMILARITY FOR ALL WORDS IN THE VOCABULARY
        similarities = [1 - textdistance.Jaccard(qval=2).distance(v, input_word) for v in word_freq_dict.keys()]
        
        # CREATE A DATAFRAME WITH THE WORDS, THEIR PROBABILITIES, AND SIMILARITY SCORES
        df = pd.DataFrame.from_dict(probs, orient='index').reset_index()
        df = df.rename(columns={'index': 'Word', 0: 'Prob'})
        df['Similarity'] = similarities
        
        # SORT THE RESULTS BASED ON SIMILARITY AND PROBABILITY, THEN RETURN THE TOP_N RESULTS
        result = df.sort_values(['Similarity', 'Prob'], ascending=False).head(top_n)[['Word', 'Similarity']].to_dict('records')
    
    # CACHE THE RESULT WITH THE CURRENT TIME
    cache[input_word] = (result, time.time())
    
    return result


# FUNCTION TO CAPITALIZE THE FIRST LETTER AFTER A PERIOD
def capitalize_sentence(sentence):
    sentence = sentence.strip()
    if not sentence:
        return sentence
    # CAPITALIZE THE FIRST LETTER OF THE SENTENCE
    sentence = sentence[0].upper() + sentence[1:]
    # CAPITALIZE THE FIRST LETTER AFTER A FULL STOP, QUESTION MARK, OR EXCLAMATION MARK
    sentence = re.sub(r'([.!?]\s+)(\w)', lambda x: x.group(1) + x.group(2).upper(), sentence)
    return sentence

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/autocorrect', methods=['POST'])
def autocorrect():
    data = request.get_json()
    input_word = data.get('word', '')
    if not input_word:
        return jsonify({'error': 'No word provided'}), 400
    results = autocorrect_word(input_word)
    suggestions = [{"Word": result["Word"], "Similarity": f"{result['Similarity']:.2f}"} for result in results]
    return jsonify({'corrections': suggestions})

@app.route('/autocorrect_sentence', methods=['POST'])
def autocorrect_sentence():
    data = request.get_json()
    sentence = data.get('sentence', '')
    if not sentence:
        return jsonify({'error': 'No sentence provided'}), 400
    
    # SEPARATE WORDS AND PUNCTUATIO
    words_in_sentence = re.findall(r'\w+|[^\w\s]', sentence)
    corrected_words = []

    for word in words_in_sentence:
        # IF THE WORD IS A PUNCTUATION, LEAVE IT AS IT IS
        if re.match(r'[^\w\s]', word):
            corrected_words.append(word)
        else:
            corrections = autocorrect_word(word, top_n=1)
            corrected_words.append(corrections[0]['Word'] if corrections else word)

    corrected_sentence = ' '.join(corrected_words)
    
    # CAPITALIZE THE SENTENCE PROPERLY
    corrected_sentence = capitalize_sentence(corrected_sentence)
    
    return jsonify({'original_sentence': sentence, 'corrected_sentence': corrected_sentence})

if __name__ == '__main__':
    app.run(debug=True)
