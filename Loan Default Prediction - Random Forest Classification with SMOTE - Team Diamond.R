# Installing Required Packages
if(!require('PerformanceAnalytics')) {
  install.packages('PerformanceAnalytics')
}
if(!require('pROC')) {
  install.packages('pROC')
}
if(!require('corrplot')) {
  install.packages('corrplot')
}
install.packages('remotes')
remotes::install_github("cran/DMwR")

#Load the required libraries
library(dplyr)
library(car)
library(PerformanceAnalytics)
library("caret")
library(corrplot)
library(tidyverse)
library(class)
library(randomForest)
library(caret)
library(e1071)
library("DMwR")
library(pROC)
options(max.print=1000000)
train <- read.csv(file.choose(), header=TRUE, sep = ",") # Please load the Training Data spreadsheet here
professionsGroupings <- read.csv(file.choose(), header=TRUE, sep = ",") # Please load the Professions Groupings spreadsheet here

#Finding the variables which have NAs
names(which(colSums(is.na(train))>0))

#Removing unnecessary characters
train$CITY <- gsub('[0-9]+', '', train$CITY)
train$CITY <- gsub("[[]]", "", train$CITY)

train$STATE <- gsub('[0-9]+', '', train$STATE)
train$STATE <- gsub("[[]]", "", train$STATE)

#creating the modified dataset
train_modified <- train

#-------------------------------
#  Transforming City and State
#-------------------------------

#Adding a variable for that combines city and state then grouping by incoming to create the rank value
train_modified$CITY_STATE <- paste(train_modified$CITY,".",train_modified$STATE)

Group_Salaries <- train_modified %>% group_by(CITY_STATE) %>% dplyr::summarize(mean = mean(Income), median = median(Income), sum = sum(Income), counts=n())
Group_Salaries$diff <- Group_Salaries$mean - Group_Salaries$median
Group_Salaries <- arrange(Group_Salaries, desc(median))
Group_Salaries$CITY_RANK <- ""

# Dividing the rank into 3 group, top 100 rows = 1, bottom 100 rows = 3, and the middle = 2
Group_Salaries$CITY_RANK[1:100] <- "1"
Group_Salaries$CITY_RANK[218:317] <- "3"
Group_Salaries$CITY_RANK[Group_Salaries$CITY_RANK==""] <- "2"

# removing unnecessary that is not needed when merging the Group_Salaries data with the train_modified data
col_drop <- c("mean", "median", "sum", "counts", "diff")
Group_Salaries = Group_Salaries[,!(names(Group_Salaries) %in% col_drop)]

# merging the Group_Salaries data with the train_modified data
train_modified <- (merge(train_modified, Group_Salaries, by = 'CITY_STATE'))
train_modified <- arrange(train_modified, Id)
train_modified$Income <- train$Income

#-------------------------------
#  Transforming profession
#-------------------------------

#adding the prefessions grouping with a merge between train_modified and professionsGroupings
train_modified <- (merge(train_modified, professionsGroupings, by = 'Profession'))
train_modified <- arrange(train_modified, Id)

# grouping the professionsGroupings by income
Group_Professions <- train_modified %>% group_by(Group1) %>% dplyr::summarize(mean = mean(Income), median = median(Income), sum = sum(Income), counts=n())
Group_Professions$diff <- Group_Professions$mean - Group_Professions$median
Group_Professions <- arrange(Group_Professions, desc(median))

# Dividing the ranking the professions group by income
Group_Professions$Profession_RANK <- row.names(Group_Professions)

# removing unnecessary that is not needed when merging the Group_Salaries data with the train_modified data
col_drop <- c("mean", "median", "sum", "counts", "diff")
Group_Professions = Group_Professions[,!(names(Group_Professions) %in% col_drop)]

# merging the Group_Salaries data with the train_modified data
train_modified <- (merge(train_modified, Group_Professions, by = 'Group1'))
train_modified <- arrange(train_modified, Id)

#load data
LoanDefault <-train_modified

LoanDefault$STATE <- NULL
Id <- LoanDefault$Id
Married.Single <- LoanDefault$Married.Single 
House_Ownership <- LoanDefault$House_Ownership
Car_Ownership <- LoanDefault$Car_Ownership
Profession <- LoanDefault$Profession
CITY <- LoanDefault$CITY
Risk_Flag <- LoanDefault$Risk_Flag
Group1 <- LoanDefault$Group1
CITY_STATE <- LoanDefault$CITY_STATE
Group2 <- LoanDefault$Group2 
CITY_RANK <- LoanDefault$CITY_RANK
Profession_RANK <- LoanDefault$Profession_RANK

# we want to exclude some of the variables in our dataset 
nonvars = c("Id","Married.Single", "House_Ownership", "Car_Ownership", "Profession",
            "CITY","Risk_Flag","Group1","CITY_STATE","Group2","CITY_RANK","Profession_RANK")
LoanDefault = LoanDefault[ , !(names(LoanDefault) %in% nonvars) ]

##Checking if Log Transformation is required
abs(skewness(LoanDefault$Income))
abs(skewness(LoanDefault$Age))
abs(skewness(LoanDefault$Experience))
abs(skewness(LoanDefault$CURRENT_JOB_YRS))
abs(skewness(LoanDefault$CURRENT_HOUSE_YRS))
#No Log Transformation required as there are no lognormal distributions

#Data Preprocessing
PreprocessLoanDefault <- preProcess(LoanDefault, method=c("center", "scale"))
LoanDefault <- predict(PreprocessLoanDefault, LoanDefault)

#Checking correlations
res <- cor(LoanDefault)
round(res, 2)
corrplot(res, type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45)
cor(LoanDefault$Experience,LoanDefault$CURRENT_JOB_YRS)
#Experience and CURRENT_JOB_YRS have high correlation between each other

#Adding back categorical variables and CITY_RANK, PROFESSION_RANK
LoanDefault <- cbind(Id,LoanDefault,Married.Single,House_Ownership,Car_Ownership,Profession
                     ,CITY,Group1,CITY_STATE,Group2,CITY_RANK,Profession_RANK,Risk_Flag)

LoanDefault$Group1 <- NULL
LoanDefault$Profession <- NULL
LoanDefault$CITY_STATE <- NULL
LoanDefault$CITY <- NULL
LoanDefault$Group2 <- NULL

#Converting categorical variables to Dummy Variables through One Hot encoding
new_data <- dummyVars("~ .", LoanDefault)
LoanDefault <- data.frame(predict(new_data, LoanDefault))

#Converting Risk_Flag to a factor for Random Forest Classification
LoanDefault$Risk_Flag <- as.factor(LoanDefault$Risk_Flag)

# Diving dataset into train and test
sample <- sample.int(n = nrow(LoanDefault), size=floor(.8*nrow(LoanDefault)), replace = F)
train <- LoanDefault[sample,]
test <- LoanDefault[-sample,]

# Remove Idvariable from training and testing sets because it is no loner useful
#in building the model
nonvars = c("Id")
train = train[ , !(names(train) %in% nonvars) ]
test = test[ , !(names(train) %in% nonvars) ]

# build a Random Forest Classifier Model to predict Risk_Flag
data_train <- train
data_test <- test

set.seed(1234)
# Define the control

#Applying SMOTE on train_data and performing RFC
#mtry value of 13 was obtained through the random forest model
#We have input the mtry value directly here to save time 
#in running the code
new_data_train <- SMOTE(Risk_Flag ~ .-CURRENT_JOB_YRS, 
                        data_train, perc.over = 100)
rf_fit <- randomForest(Risk_Flag ~.-CURRENT_JOB_YRS, 
                       data = new_data_train,
                       mtry=13, maxnodes = NULL,
                       importance = TRUE )
print(rf_fit)

prediction <-predict(rf_fit, data_test)
confusionMatrix(prediction, data_test$Risk_Flag)
prediction <- as.numeric(prediction)
auc(data_test$Risk_Flag,prediction)

#Importance Plots of the RFC model with SMOTE
varImpPlot(rf_fit)
importance(rf_fit)