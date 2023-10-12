# Verificando o diretório de trabalho atual
getwd()
setwd("D:\\Inteligência Artificial\\Repositórios\\titanic_crash")
getwd()

# Importando o dataset
data.frame <- read.csv("train.csv", na.strings="")

# Instalando os pacotes necessários
install.packages('psych')
install.packages('GGally')
install.packages('dplyr')
install.packages('ggplot2')
install.packages('rpart')
install.packages('rpart.plot')
install.packages('Amelia')
install.packages('e1071')
install.packages('dummies')

# Limpando e preparando o dataset
View(data.frame)

library(Amelia)
missmap(data.frame, col=c('black', 'grey'))

library(dplyr)
data.frame = select(data.frame, Survived, Pclass, Age, Sex, SibSp, Parch)

data.frame = na.omit(data.frame)

str(data.frame)

library(GGally)
ggcorr(data.frame,
       nbreaks = 6,
       label = TRUE,
       label_size = 3,
       color = 'grey50')

data.frame$Survived = factor(data.frame$Survived)
data.frame$Pclass = factor(data.frame$Pclass, order=TRUE, levels = c(3, 2, 1))

str(data.frame)

# Visualizando os dados
library(ggplot2)
ggplot(data.frame, aes(x = Survived)) +
  geom_bar(width=0.5, fill = 'coral') +
  geom_text(stat='count', aes(label=stat(count)), vjust=-0.5) +
  theme_classic()

ggplot(data.frame, aes(x = Survived, fill=Sex)) +
  geom_bar(position = position_dodge()) +
  geom_text(stat='count',
            aes(label=stat(count)), 
            position = position_dodge(width=1), vjust=-0.5)+
  theme_classic()

ggplot(data.frame, aes(x = Survived, fill=Pclass)) +
  geom_bar(position = position_dodge()) +
  geom_text(stat='count',
            aes(label=stat(count)), 
            position = position_dodge(width=1), 
            vjust=-0.5)+
  theme_classic()

ggplot(data.frame, aes(x = Age)) +
  geom_density(fill='coral')

data.frame$Discretized.age = cut(data.frame$Age, c(0,10,20,30,40,50,60,70,80,100))
ggplot(data.frame, aes(x = Discretized.age, fill=Survived)) +
  geom_bar(position = position_dodge()) +
  geom_text(stat='count', aes(label=stat(count)), position = position_dodge(width=1), vjust=-0.5)

# Criando dataset de treino e de teste
train_test_split = function(data, fraction = 0.8, train = TRUE) {
  total_rows = nrow(data)
  train_rows = fraction * total_rows
  sample = 1:train_rows
  if (train == TRUE) {
    return (data[sample, ])
  } else {
    return (data[-sample, ])
  }
}

train <- train_test_split(data.frame, 0.8, train = TRUE)
test <- train_test_split(data.frame, 0.8, train = FALSE)

# Decision Tree
library(rpart)
library(rpart.plot)
fit <- rpart(Survived~., data = train, method = 'class')
rpart.plot(fit, extra = 106)

# Acurácia
predicted = predict(fit, test, type = 'class')
table = table(test$Survived, predicted)
dt_accuracy = sum(diag(table)) / sum(table)
paste("The accuracy is : ", dt_accuracy)

## Fine Tuning
control = rpart.control(minsplit = 8,
                        minbucket = 2,
                        maxdepth = 6,
                        cp = 0)
tuned_fit = rpart(Survived~., data = train, method = 'class', control = control)
dt_predict = predict(tuned_fit, test, type = 'class')
table_mat = table(test$Survived, dt_predict)
dt_accuracy_2 = sum(diag(table_mat)) / sum(table_mat)
paste("The accuracy is : ", dt_accuracy_2)

# Logistic Regression
data_rescale = mutate_if(data.frame,
                         is.numeric,
                         list(~as.numeric(scale(.))))
r_train = train_test_split(data_rescale, 0.7, train = TRUE)
r_test = train_test_split(data_rescale, 0.7, train = FALSE)
logit = glm(Survived~., data = r_train, family = 'binomial')
summary(logit)
lr_predict = predict(logit, r_test, type = 'response')
table_mat = table(r_test$Survived, lr_predict > 0.68)
lr_accuracy = sum(diag(table_mat)) / sum(table_mat)
paste("The accuracy is : ", lr_accuracy)

# Naive Bayes
library(e1071)
nb_model = naiveBayes(Survived ~., data=train)
nb_predict = predict(nb_model,test)
table_mat = table(nb_predict, test$Survived)
nb_accuracy = sum(diag(table_mat)) / sum(table_mat)
paste("The accuracy is : ", nb_accuracy)

# KNN
library(class)
library(dummies)

ohdata = cbind(data.frame, dummy(data.frame$Pclass))
ohdata = cbind(ohdata, dummy(ohdata$Sex))

ohdata$Pclass = NULL
ohdata$Sex = NULL
ohtrain = train_test_split(ohdata, 0.8, train = TRUE)
ohtest = train_test_split(ohdata, 0.8, train = FALSE)
train_labels = select(ohtrain, Survived)[,1]
test_labels = select(ohtest, Survived)[,1]

ohtrain$Survived=NULL
ohtest$Survived=NULL
knn_predict = knn(train = ohtrain,
                  test = ohtest,
                  cl = train_labels,
                  k=10)
table_mat = table(knn_predict, test_labels)
accuracy_knn = sum(diag(table_mat)) / sum(table_mat)