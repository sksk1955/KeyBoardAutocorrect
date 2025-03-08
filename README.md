# Jaccard Similarity-Based NLP AutoCorrect

## Overview
This project implements an NLP-based autocorrect system using Jaccard similarity and K-Nearest Neighbors (KNN) for text correction. It leverages Python, Flask for the backend, and HTML, CSS for the frontend to provide an interactive autocorrect feature.

## Features
- **Jaccard Similarity-based Autocorrect**: Computes similarity scores for words and suggests corrections.
- **KNN-based Autocorrect (Commented out)**: Uses edit distance for better word prediction.
- **Probability-based Word Correction**: Suggests words based on frequency in a given dataset.
- **Interactive Web Interface**: Built with Flask, HTML, and CSS.
- **Handles Full Sentence Corrections**: Corrects entire sentences while preserving punctuation.

## Technologies Used
- **Backend**: Python, Flask
- **Frontend**: HTML, CSS
- **Libraries**: Pandas, NumPy, TextDistance, Collections, Re

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/-repo_name-
   cd autocorrect-nlp
   ```
2. Install dependencies:
   ```sh
   pip install flask pandas numpy textdistance
   ```
3. Run the Flask server:
   ```sh
   python app.py
   ```
4. Open the web interface in your browser:
   ```
   http://127.0.0.1:5000/
   ```

## Usage
- Enter a sentence in the text input field.
- The system will suggest corrected words and display the best possible corrected sentence.
- For custom datasets, replace `a1.csv` with your word list.

## File Structure
```
├── app.py                  # Flask backend
├── templates/
│   ├── index.html          # Frontend UI          # Styling
├── Oxford.txt                  # Word dataset
├── model.py          # Core logic for Jaccard-based correction
├── README.md               # Project documentation
```


_Developed by SaumyaSKala_

