# SMS Spam Collection Classification

## Project Overview
This project demonstrates the process of building a machine learning model to classify SMS messages as either 'ham' (legitimate) or 'spam'. It utilizes Natural Language Processing (NLP) techniques, specifically Bag of Words (BOW) and TF-IDF (Term Frequency-Inverse Document Frequency), for text feature extraction, followed by a Multinomial Naive Bayes classifier for model training and prediction.

## Dataset
The dataset used is the "SMS Spam Collection" which contains a set of SMS messages tagged as either 'ham' or 'spam'.

## Methodology
The project follows these key steps:

1.  **Data Loading and Initial Exploration**: The SMS messages and their labels are loaded into a pandas DataFrame.
2.  **Text Preprocessing and Cleaning**: Raw SMS messages undergo several preprocessing steps:
    *   Removal of non-alphabetic characters.
    *   Conversion to lowercase.
    *   Tokenization and stemming using `PorterStemmer`.
    *   Removal of common English stopwords.
    The cleaned text is compiled into a `corpus`.
3.  **Label Encoding**: The 'label' column ('ham', 'spam') is converted into numerical format (0, 1) using one-hot encoding.
4.  **Train-Test Split**: The dataset is split into training and testing sets to evaluate model performance on unseen data.
5.  **Feature Engineering (Bag of Words - BOW)**:
    *   `CountVectorizer` is used to convert the preprocessed text into a matrix of token counts. `max_features` is set to 2500, and `ngram_range` is set to (1,2) to include unigrams and bigrams.
    *   A Multinomial Naive Bayes model is trained on the BOW features.
    *   The model's accuracy and a classification report are generated.
6.  **Feature Engineering (TF-IDF)**:
    *   `TfidfVectorizer` is used to transform the preprocessed text into TF-IDF features. Similar to BOW, `max_features` is set to 2500, and `ngram_range` is (1,2).
    *   A separate Multinomial Naive Bayes model is trained on the TF-IDF features.
    *   The model's accuracy and a classification report are generated.

## Results

### Bag of Words (BOW) Model Performance
*   **Accuracy**: Approximately 98.7%
*   **Classification Report**:
    ```
                   precision    recall  f1-score   support

         False       0.94      0.96      0.95       156
          True       0.99      0.99      0.99       959

      accuracy                           0.99      1115
     macro avg       0.97      0.98      0.97      1115
  weighted avg       0.99      0.99      0.99      1115
    ```

### TF-IDF Model Performance
*   **Accuracy**: Approximately 97.7%
*   **Classification Report**:
    ```
                   precision    recall  f1-score   support

         False       0.99      0.84      0.91       158
          True       0.97      1.00      0.99       957

      accuracy                           0.98      1115
     macro avg       0.98      0.92      0.95      1115
  weighted avg       0.98      0.98      0.98      1115
    ```

## Conclusion
Both Bag of Words and TF-IDF feature extraction methods, coupled with a Multinomial Naive Bayes classifier, demonstrate high accuracy in distinguishing between ham and spam messages. The BOW model slightly outperforms the TF-IDF model in this particular setup.

## Dependencies
*   `pandas`
*   `nltk`
*   `scikit-learn`
*   `numpy`

To run this notebook, ensure you have these libraries installed. NLTK stopwords need to be downloaded (`nltk.download('stopwords')`).
