##################################################################################
# IMPLEMENTATION OF JACCARD SIMILARITY BASED NLP MODEL BASED AUTOCORRECT
##################################################################################
import pandas as pd
import numpy as np
import textdistance  # FOR STRING SIMILARITY METRICS
import re
from collections import Counter

# READ THE DATA FROM THE "Oxford.txt" FILE AND CLEAN IT
words = []
with open("a1.csv", encoding="utf8") as f:
    file_name_data = f.read()
    file_name_data = file_name_data.lower()  # CONVERT ALL WORDS TO LOWERCASE
    words = re.findall(r"\w+", file_name_data)  # EXTRACT ALL WORDS USING REGULAR EXPRESSIONS

# CREATE A SET OF UNIQUE WORDS
V = set(words)
print(f"The first ten words in the text are: \n{words[0:10]}")
print(f"There are {len(V)} unique words in the vocabulary")

# COUNT THE FREQUENCY OF EACH WORD USING COUNTER
word_freq_dict = Counter(words)
print(word_freq_dict.most_common()[0:10])

# CALCULATE THE PROBABILITY DISTRIBUTION OF EACH WORD IN THE VOCABULARY
probs = {}
Total = sum(word_freq_dict.values())  # TOTAL WORD COUNT
for k in word_freq_dict.keys():
    probs[k] = word_freq_dict[k] / Total  # PROBABILITY = FREQUENCY / TOTAL

# DEFINE THE FIRST AUTOCORRECT MODEL USING JACCARD SIMILARITY
def autocorrect_word(input_word, top_n=5):
    input_word = input_word.lower()
    
    # IF THE WORD IS ALREADY IN THE VOCABULARY, RETURN IT AS IS
    if input_word in V:
        return pd.DataFrame({'Word': [input_word], 'Similarity': [1.0]})
    
    # CALCULATE JACCARD SIMILARITY WITH EACH WORD IN THE VOCABULARY
    similarities = [1 - (textdistance.Jaccard(qval=2).distance(v, input_word)) for v in word_freq_dict.keys()]
    
    # CREATE A DATAFRAME TO STORE PROBABILITIES AND SIMILARITIES
    df = pd.DataFrame.from_dict(probs, orient='index').reset_index()
    df = df.rename(columns={'index': 'Word', 0: 'Prob'})
    df['Similarity'] = similarities  # ADD SIMILARITY SCORES TO THE DATAFRAME
    
    # SORT WORDS BY SIMILARITY AND PROBABILITY, SELECTING THE TOP-N RESULTS
    output = df.sort_values(['Similarity', 'Prob'], ascending=False).head(top_n)
    
    return output[['Word', 'Similarity']]

# AUTOCORRECT A FULL SENTENCE USING THE JACCARD MODEL
def autocorrect_sentence(sentence, top_n=5):
    sentence = sentence.lower()
    words_with_punct = re.findall(r'\w+|[^\w\s]', sentence)  # SEPARATE WORDS AND PUNCTUATION
    
    corrections = {}
    corrected_sentence = []
    
    for word in words_with_punct:
        if re.match(r'\w+', word):  # IF THE WORD CONTAINS ALPHANUMERIC CHARACTERS
            corrected_words = autocorrect_word(word, top_n)
            corrections[word] = corrected_words
            best_word = corrected_words.iloc[0, 0]  # PICK THE BEST MATCH
            corrected_sentence.append(best_word)
        else:
            corrected_sentence.append(word)  # PRESERVE PUNCTUATION
    
    final_sentence = ' '.join(corrected_sentence).capitalize()  # CAPITALIZE THE FINAL OUTPUT
    return corrections, final_sentence

# INPUT SENTENCE FOR JACCARD MODEL
input_sentence = "Thee guillty must be cauught annd be subjekted to interogetion."
corrections, best_sentence = autocorrect_sentence(input_sentence, top_n=5)

print(f"Original sentence: {input_sentence}")
for word, correction_df in corrections.items():
    print(f"\nPossible corrections for '{word}':")
    print(correction_df)
    
print(f"\nThe corrected sentence: {best_sentence}")
#######################################################################################
# IMPLEMENTATION OF K-NEAREST NEIGHBOR (KNN)-BASED AUTOCORRECT
#######################################################################################
# import pandas as pd
# import numpy as np
# import re
# from collections import Counter

# # PREPROCESS DATA SIMILARLY AS ABOVE
# words = []
# with open("Oxford.txt", encoding="utf8") as f:
#     file_name_data = f.read()
#     file_name_data = file_name_data.lower()
#     words = re.findall(r"\w+", file_name_data)

# Vocabulary = set(words)
# print(f"The first ten words in the text are: \n{words[0:10]}")
# print(f"There are {len(Vocabulary)} unique words in the vocabulary")

# WordFreqDictionary = Counter(words)
# print(WordFreqDictionary.most_common()[0:10])

# # CALCULATE PROBABILITIES
# probs = {}
# Total = sum(WordFreqDictionary.values())
# for k in WordFreqDictionary.keys():
#     probs[k] = WordFreqDictionary[k] / Total

# # IMPLEMENT EDIT DISTANCE FUNCTION
# def calculate_distance(word1, word2):
#     """Calculate the edit distance between two words."""
#     m, n = len(word1), len(word2)
#     dp = [[0] * (n + 1) for _ in range(m + 1)]
#     for i in range(m + 1):
#         for j in range(n + 1):
#             if i == 0:
#                 dp[i][j] = j
#             elif j == 0:
#                 dp[i][j] = i
#             elif word1[i - 1] == word2[j - 1]:
#                 dp[i][j] = dp[i - 1][j - 1]
#             else:
#                 dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
#     return dp[m][n]

# # KNN AUTOCORRECT FUNCTION
# def knn_autocorrect(input_word, top_n=5):
#     input_word = input_word.lower()
    
#     # IF THE WORD IS ALREADY IN THE VOCABULARY, RETURN IT AS IS
#     if input_word in Vocabulary:
#         return pd.DataFrame({'Word': [input_word], 'Similarity': [1.0]})
    
#     distances = []
#     for vocab_word in WordFreqDictionary.keys():
#         distance = calculate_distance(input_word, vocab_word)  # CALCULATE EDIT DISTANCE
#         distances.append((vocab_word, distance))
    
#     # SORT BY DISTANCE AND PROBABILITY, SELECTING THE TOP-N RESULTS
#     sorted_words = sorted(distances, key=lambda x: (x[1], -probs[x[0]]))[:top_n]
#     similarity_scores = [1 - (dist / max(len(input_word), len(vocab_word))) for vocab_word, dist in sorted_words]
#     df = pd.DataFrame(sorted_words, columns=['Word', 'Distance'])
#     df['Similarity'] = similarity_scores  # CALCULATE SIMILARITY SCORES
    
#     return df[['Word', 'Similarity']]

# # KNN AUTOCORRECT SENTENCE FUNCTION
# def knn_autocorrect_sentence(sentence, top_n=5):
#     sentence = sentence.lower()
#     words_in_sentence = re.findall(r'\w+', sentence)  # SPLIT INTO WORDS
    
#     corrections = {}
#     best_sentence = []
    
#     for word in words_in_sentence:
#         corrected_words = knn_autocorrect(word, top_n)
#         corrections[word] = corrected_words
#         best_word = corrected_words.iloc[0, 0]  # PICK THE BEST MATCH
#         best_sentence.append(best_word)
    
#     return corrections, ' '.join(best_sentence)

# # INPUT SENTENCE FOR KNN MODEL
# input_sentence = "I lovvei the smmell afteer it raiins"
# corrections, best_sentence = knn_autocorrect_sentence(input_sentence, top_n=5)

# print(f"Original sentence: {input_sentence}")
# for word, correction_df in corrections.items():
#     print(f"\nPossible corrections for '{word}':")
#     print(correction_df)
# print("\n******************************************")
# print(f"\nThe corrected sentence: {best_sentence}")
# print("\n******************************************")

