1. Upload the provided data file to your repository and provide a link to the uploaded file in your report. (1 point). - https://github.com/EleneBa/aimlmid2026_e_bakuradze25/blob/main/Task2/Data/e_bakuradze25_23456_csv.csv

2. Dataset description:
The dataset contains numeric features extracted from email texts.
The features include total word count (words), number of hyperlinks (links), number of capitalized words (capital_words), and count of known spam-related words (spam_word_count).
The target variable is_spam indicates whether the email is spam (1) or legitimate (0).

https //github.com/EleneBa/aimlmid2026_e_bakuradze25/blob/main/Task2/src/train_eval.py

Model used (with code) for logistic regression - 
model = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=2000))
])

Provide the coefficients found - 
=== Model coefficients ===
Intercept: 1.9967161430216218
Feature coefficients:
capital_words: 3.599764145013232
spam_word_count: 2.2316882394172026
links: 2.1693102548113394
words: 1.6923756456106873

=== Evaluation on 30% holdout ===
Confusion Matrix:
[[364   9]
 [ 20 357]]
Accuracy: 0.961333

