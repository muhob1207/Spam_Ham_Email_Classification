library(data.table)
library(tidyverse)
library(text2vec)
library(caTools)
library(glmnet)


#Q1. Import emails dataset and get familiarized with it.
data <- fread('emails.csv')
data %>% view()

#Q2. Add `id` column defined by number of rows
data$id <- seq(1:nrow(data)) %>% as.character()

#Q3. Prepare data for fitting to the model

# Split data
set.seed(123)
split <- data$spam %>% sample.split(SplitRatio = 0.8)
train <- data %>% subset(split == T)
test <- data %>% subset(split == F)

#Creating the vocabulary and vectorizing the words
train_tokens <- train$text %>% tolower() %>% word_tokenizer()

it_train <- train_tokens %>% 
  itoken(ids = train$id,
         progressbar = F)

vocab <- it_train %>% create_vocabulary()

vocab %>% 
  arrange(desc(term_count)) %>% 
  head(10)

vectorizer <- vocab %>% vocab_vectorizer()
dtm_train <- it_train %>% create_dtm(vectorizer)

dtm_train %>% dim()
identical(rownames(dtm_train), train$id)

#Q4. Use cv.glmnet for modeling
glmnet_classifier <- dtm_train %>% 
  cv.glmnet(y = train[['spam']],
            family = 'binomial', 
            type.measure = "auc",
            nfolds = 10,
            thresh = 0.001,# high value is less accurate, but has faster training
            maxit = 1000)# again lower number of iterations for faster training

#Q5. Give interpretation for train and test results

#For train data the maximum AUC is 0.994
glmnet_classifier$cvm %>% max() %>% paste("-> Max AUC")

#Vectorizing the test data
it_test <- test$text %>% tolower() %>% word_tokenizer()

it_test <- it_test %>% 
  itoken(ids = test$id,
         progressbar = F)

dtm_test <- it_test %>% create_dtm(vectorizer)

#Making predictions for test data
preds <- predict(glmnet_classifier, dtm_test, type = 'response')[,1]

#The AUC for test data is 0.993. There is a very small degree of overfitting.
glmnet:::auc(test$spam, preds) 
