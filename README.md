# 📩 SMS Spam Classifier

🚀 **Overview**
This project implements an SMS spam classifier using the **Naïve Bayes algorithm** in R. The model is trained on a dataset of spam and ham messages and optimized for accuracy.

## 📂 Project Structure
```
📦 SMS-Spam-Classifier
├── 📄 README.md  
├── 📂 data  
│   ├── spam.csv  
├── 📂 src  
│   ├── preprocessing.R  
│   ├── train_model.R  
│   ├── evaluate_model.R  
├── 📂 notebooks  
│   ├── sms_classification.ipynb  
├── 📂 reports  
│   ├── results.md  
└── 📂 visualization  
    ├── wordcloud.png  
```

## 📊 Data Cleaning and Preprocessing
1. **Import Data** 📥
```r
sms_raw <- read.csv("data/spam.csv", stringsAsFactors = FALSE)
str(sms_raw)
sms_raw$type <- factor(sms_raw$type)
```

2. **Text Processing** 🔍
```r
sms_corpus <- Corpus(VectorSource(sms_raw$text))
corpus_clean <- tm_map(sms_corpus, tolower)
corpus_clean <- tm_map(corpus_clean, removeNumbers)
corpus_clean <- tm_map(corpus_clean, removeWords, stopwords())
corpus_clean <- tm_map(corpus_clean, removePunctuation)
corpus_clean <- tm_map(corpus_clean, stripWhitespace)
sms_dtm <- DocumentTermMatrix(corpus_clean)
```

## 🎨 Visualization
- **Word Cloud** ☁️
```r
install.packages("wordcloud")
library(wordcloud)
wordcloud(sms_corpus, min.freq = 40, random.order = FALSE)
```
- **Spam vs Ham Word Clouds** 📊
```r
spam <- subset(sms_raw, type == "spam")
ham <- subset(sms_raw, type == "ham")
wordcloud(spam$text, max.words = 40, scale = c(3, 0.5))
wordcloud(ham$text, max.words = 40, scale = c(3, 0.5))
```

## 🤖 Model Training
1. **Feature Engineering** 🛠️
```r
convert_counts <- function(x) {
 x <- ifelse(x > 0, 1, 0)
 x <- factor(x, levels = c(0, 1), labels = c("No", "Yes"))
 return(x)
}
sms_train <- apply(sms_dtm, MARGIN = 2, convert_counts)
```
2. **Naïve Bayes Classifier** ⚡
```r
install.packages("e1071")
library(e1071)
sms_classifier <- naiveBayes(sms_train, sms_raw$type)
```

## 📈 Model Evaluation
```r
sms_test_pred <- predict(sms_classifier, sms_test)
CrossTable(sms_test_pred, sms_raw$type, prop.chisq = FALSE, prop.t = FALSE, dnn = c('Predicted', 'Actual'))
```

## 🚀 Optimizing Performance
```r
sms_classifier2 <- naiveBayes(sms_train, sms_raw$type, laplace = 0.1)
sms_test_pred2 <- predict(sms_classifier2, sms_test)
CrossTable(sms_test_pred2, sms_raw$type, prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE, dnn = c('Predicted', 'Actual'))
```
🔹 **False positives reduced from 5 to 2!**
🔹 **False negatives reduced from 26 to 20!**

## 🎯 Conclusion
This project demonstrates the effectiveness of the **Naïve Bayes classifier** in text classification, particularly for **SMS spam detection**. 🚀

### 🌟 Author
📌 **Oumayma Abayed** | Data Science Enthusiast


