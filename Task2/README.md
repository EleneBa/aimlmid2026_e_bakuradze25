**1.** Upload the provided data file to your repository and provide a link to the uploaded file in your report. (1 point). - https://github.com/EleneBa/aimlmid2026_e_bakuradze25/blob/main/Task2/Data/e_bakuradze25_23456_csv.csv

**2. **Dataset description:****
The dataset contains numeric features extracted from email texts.
The features include total word count (words), number of hyperlinks (links), number of capitalized words (capital_words), and count of known spam-related words (spam_word_count).
The target variable is_spam indicates whether the email is spam (1) or legitimate (0).

**Data Loading and Processing**
The dataset is loaded from a CSV file using pandas. The label column (is_spam) is separated from the feature columns. All feature values are numeric and are used directly for model training. The dataset is split into training and validation sets using a 70% / 30% split, with stratification to preserve class balance.

**Source code:**
https://github.com/EleneBa/aimlmid2026_e_bakuradze25/blob/main/task2_spam/src/train_eval.py

**Model used (with code) for logistic regression** - 
model = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=2000))
])

**Provide the coefficients found** - 
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
 
**3. Model Validation on Unused Data**
Confusion Matrix and Accuracy
The trained model was evaluated on the 30% holdout (unused) data.

Confusion Matrix:
[[364   9]
 [ 20 357]]

**Explanation:**
True Negatives (TN) = 364 → non-spam correctly classified as non-spam
False Positives (FP) = 9 → non-spam incorrectly classified as spam
False Negatives (FN) = 20 → spam incorrectly classified as non-spam
True Positives (TP) = 357 → spam correctly classified as spam
The confusion matrix summarizes the model’s classification performance by counting correct and incorrect predictions across both classes.

Accuracy = (357 + 364) / (357 + 364 + 9 + 20) = 0.961333

4. I generated a spam email by maximizing the feature values that the trained model associates with spam. Based on the learned coefficients, capital_words and spam_word_count are strong positive indicators, so I used many ALL-CAPS words (e.g., URGENT, WINNER, FREE, PRIZE) and repeated common spam keywords (free, winner, claim, prize, money, offer, urgent). I also included a URL to increase the links feature, and ensured the email contains enough text to raise the overall words count. These choices increase the weighted linear score of logistic regression, resulting in a SPAM prediction and a higher spam probability.


**the command and outcome **-
(.venv) C:\Users\EleneBakuradze\CDA01\aimlmid2026_e_bakuradze25\task2_spam\src>python app.py
Model is trained. Paste an email text below. End with an empty line.
Subject: URGENT WINNER FREE PRIZE
YOU ARE A WINNER WINNER WINNER
CLAIM YOUR FREE PRIZE MONEY NOW
LIMITED OFFER BONUS CASH NOW
CLICK http://free-prize-win.example.com
FREE FREE FREE WIN PRIZE CLAIM MONEY

Prediction: SPAM
Spam probability: 1.0000



