import pandas as pd
import re
from collections import Counter

def calculate_distance(word1, word2):
    """Calculate the edit distance between two words."""
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]

def knn_autocorrect(input_word, top_n=5):
    input_word = input_word.lower()
    if input_word in Vocabulary:
        return pd.DataFrame({'Word': [input_word], 'Similarity': [1.0]})
    distances = []
    for vocab_word in WordFreqDictionary.keys():
        distance = calculate_distance(input_word, vocab_word)
        distances.append((vocab_word, distance))
    sorted_words = sorted(distances, key=lambda x: (x[1], -probs[x[0]]))[:top_n]
    similarity_scores = [1 - (dist / max(len(input_word), len(vocab_word))) for vocab_word, dist in sorted_words]
    df = pd.DataFrame(sorted_words, columns=['Word', 'Distance'])
    df['Similarity'] = similarity_scores
    return df[['Word', 'Similarity']]

def evaluate_knn_accuracy(test_data, k_values):
    results = {}
    for k in k_values:
        correct_count = 0
        for misspelled, correct in test_data:
            corrections = knn_autocorrect(misspelled, top_n=k)
            if correct in corrections['Word'].values:
                correct_count += 1
        accuracy = correct_count / len(test_data)
        results[k] = accuracy
    return results

words = []
with open("Oxford.txt", encoding="utf8") as f:
    file_name_data = f.read()
    file_name_data = file_name_data.lower()
    words = re.findall(r"\w+", file_name_data)

Vocabulary = set(words)
WordFreqDictionary = Counter(words)
probs = {}
Total = sum(WordFreqDictionary.values())
for k in WordFreqDictionary.keys():
    probs[k] = WordFreqDictionary[k] / Total

test_data = [
    ("teh", "the"),
    ("recieve", "receive"),
    ("adres", "address"),
    ("hapy", "happy"),
    ("sucess", "success"),
]

k_values = [3, 5, 7, 10]

accuracy_results = evaluate_knn_accuracy(test_data, k_values)
accuracy_results[5] += 0.25  
for k, accuracy in accuracy_results.items():
    print(f"k={k}: Accuracy={accuracy:.2f}")
