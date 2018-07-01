# ************************************************
# Name: ChurnPredictor.R
# Written and Submitted by: Keerty Agarwal on 11th June 2018
#
# ML project code to predict telecom customers who churn
# The code is completely written in R. 
# Most of the functions have been reused from the course labs details follow
# Many libraries from CRAN are used details follow
# The code runs and then compares results of the following algorithms
# LDA
# Decision Tree
# Decision Tree with Boost
# Random Forest
# MLP Neural Network
# Deep Neural Network
#
# The usage of the models and default values have been guided by the course lab 
# and code used along with functions which was written by 
# Ryman-Tubb, N. F. (2018), Pre-processing functions in R Code. 
# The Surrey Business School, University of Surrey, 
# MSc Machine Learning & Visualisation Module
#
# *********************************************************************

# *********************************************************************
# The following functions are used from course labs thanks to Prof. Nick. These 
# have been very helpful in understand the ML algorithms and processing.
# Some functions have been used as is and some have been modified 
# to resolve the problem at hand eg. NcalcConfusion()
#
# N_setup()
# NreadDataset()
# NPREPROCESSING_removePunctuation()
# NPREPROCESSINGpretty_dataset_table()
# NPREPROCESSING_splitdataset()
# NLDAperformance()
# NROCthreshold()
# Nrescaleentireframe()
# NPREPROCESSING_redundantFields() 
# LEARN_BasicNeural()
# EVALUATE_BasicNeural()
# NROCgraph()
# N_DEEP_Initialise()
# N_DEEP_TrainClassifier()
# NcalcFPR()
# NcalcFNR()
# Nrmse()
# NcalcMeasures()
# NcalcConfusion()
# N_printMeasures()
# NPLOT_correlagram() 
#
# written by 
# Ryman-Tubb, N. F. (2018), Pre-processing functions in R Code. 
# The Surrey Business School, University of Surrey, 
# MSc Machine Learning & Visualisation Module
# ********************************************************************

# ************************************************
# Define the libraries used in this project
# plotrix     3.6-5      Plot/graphics   https://cran.r-project.org/web/packages/plotrix/plotrix.pdf
# neuralnet   1.33       Basic MLP       https://cran.r-project.org/web/packages/neuralnet/neuralnet.pdf
# outliers    0.14       Outlier detect  https://cran.r-project.org/web/packages/outliers/outliers.pdf
# h2o         3.16.0.2   Deep NN Clus.   https://cran.r-project.org/web/packages/h2o/h2o.pdf
# pROC        1.10.0     ROC Chart       https://cran.r-project.org/web/packages/pROC/pROC.pdf
# formattable
# MASS

# ************************************************
# This is where R starts execution

print("START ChurnPredictor.R")

packages_required<-c("xtable","PerformanceAnalytics","plotrix","neuralnet", "corrplot","h2o","pROC","formattable","MASS")

# Install the packages to ensure smooth running of the code
install.packages(packages_required)

library(outliers)
library(neuralnet)
library(h2o)
library(pROC)
library(formattable)
library(MASS)
library(dummies)
library(plotrix)
library(C50)
library(randomForest)
library("xtable")
library(corrplot)

# Check the current Working directory
getwd()

# set the working directory to the folder where the dependent data files and code are kept
setwd("/Users/keerty/Desktop/Coursework")

source("CP_functions.R")

#Variables which are used in the code, capital helps with standing them out

COLLIN_CUTOFF     <- 0.90                  # if correlation is >0.9 then only collinear else not
BASICNN_HIDDEN    <- 1                   # hidden layer neurons
BASICNN_THRES     <- 0.01                 #Error threshold to stop training
BASICNN_EPOCHS    <- 5                   #Maximum number of training epocs
DECISION_THR      <- 0.7                  #Manual threshold (cut-off) for decision = Churn
CLASS0            <- 0                    #Indicates "NotChurn" class
CLASS1            <- 1                    #Indicates "Churn" class
DEEP_HIDDEN       <- c(30,10)             #Number of neurons in each layer
DEEP_STOPPING     <- 15                   #Number of times no improvement before stop
DEEP_TOLERANCE    <- 1e-4                 #Error threshold
DEEP_ACTIVATION   <- "Tanh"               #Non-linear activation function
DEEP_REPRODUCABLE <- TRUE                 #Set to TRUE to test training is same for each run
COST              <- 750                  #cost to acquire a new customer as provided in $
OUTPUT_FIELD      <-"Churn"               #The name of column which contains the classifier
BOOST_DT          <- 100                  #boosting trials on a decision tree by varying values from 20 - 50 -100
FOREST_SIZE       <- 1500                 #Number of trees in the forest


#To ensure each run produces the same result
set.seed("1234")

# Load the data csv file procvided for this assignment
orig_dataset<-read.csv("Telco-Customer-Churn-MANM354.csv")

#gives the structure of the dataset a good overview to get an idea.
str(orig_dataset)

#it was noticed that SeniorCitizen was not a factor, hence changing it to a factor
orig_dataset$SeniorCitizen<-as.factor(orig_dataset$SeniorCitizen)

#rerun structure
str(orig_dataset)

#to summarise each dimension in detail
summary(orig_dataset[-1])

#study somerows of the data
head(orig_dataset)
names(orig_dataset)

#To see a presentable version of str output using a preprocessing function
NPREPROCESSINGpretty_dataset_table(orig_dataset)

#to preprocess we want columns as characters and not as factors and hence reading the dataset again with correct option
orig_dataset<-read.csv("Telco-Customer-Churn-MANM354.csv", stringsAsFactors = FALSE)

# visualisation
hist(orig_dataset$tenure, main = paste("Histogram of Tenure"),xlab="No. of Months", xlim=range(orig_dataset$tenure),col="pink")
plot(sort(orig_dataset$MonthlyCharges), orig_dataset$churn, main="Monthly charges", xlab="Row Number", ylab="Charges in $", col="red")


#Binarising fields with 2 categories to 0 and 1
# Gender Male=0, Female=1
# For Columns
# gender, Partner, Dependents, PhoneService, MultipleLines, 
# InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, 
# StreamingTV, StreamingMovies, PaperlessBilling, Churn
# No=0, Yes=1
# No internet service and No phone service =0

binCols<-c(2,4,5,7,8,10:15,17,21)

pro_data<-orig_dataset
#Looping through each column and converting binarising the data
for(i in binCols) {
  for(j in 1:nrow(pro_data)) {
    pro_data[j,i]<-ifelse(pro_data[j,i] == "Yes" ,1,ifelse(pro_data[j,i] == "Female",1,0))
  }
  pro_data[,i]<-as.numeric(pro_data[,i])
}
undum_data<-pro_data

#checking
str(undum_data)

#Histogram of gender
hist(undum_data$gender,main="Gender distribution",xlab="Gender", col = "yellow" )

undumbin_data<-undum_data

#Now is the time to binarize the categorical features i.e. factors, we omit the 1st column
binTrainingData <- dummy.data.frame(undumbin_data[,-1],sep = ".")
unscale_data<-binTrainingData

#checking
str(unscale_data)


#Unscale the tenure and monthly cost column
#omitting the total charges column as correlated to tenure and charges
unscale_data[,5]<-rescale(unscale_data[,5],range(0,1))
unscale_data[,25]<-rescale(unscale_data[,25],range(0,1))
scaled_data<-unscale_data

#Checking
str(scaled_data)

#replace the name of monthly charges column to scaled monthly charges
names(scaled_data)[names(scaled_data) == "MonthlyCharges"] <- "ScaledMonthlyCharges"

#Checking
str(scaled_data)

#combining the actual monthlycosts to the scaled dataset as required for future calculations
combinedforML<-cbind(scaled_data[-26],orig_dataset[c(19,1)])

#checking
str(combinedforML)

#colnames contain punctuation, to clean the same
names(combinedforML)<-NPREPROCESSING_removePunctuation(names(combinedforML))

#checking
names(combinedforML)

#Checking for any redundant fields, collinearity/ repeated values
clean_dataset<-NPREPROCESSING_redundantFields(combinedforML[,1:26],COLLIN_CUTOFF) 

#nothing got deleted so no collinearity/ redundancy
str(clean_dataset)

#split dataset in training and test (70:30)
r<-NPREPROCESSING_splitdataset(clean_dataset)

#train data: dataframe with just the input fields
# train_inputs<-r$train[,!names(r$train)==c(OUTPUT_FIELD,"MonthlyCharges","customerID")]
train_inputs<-r$train[,1:25]
# str(train_inputs)

#train data: vector with the expected output class
train_expected<-r$train[,names(r$train)==OUTPUT_FIELD]

#test data: dataframe with just the input fields
test_inputs<-r$test[,1:25]

#test data: vector with just the expected output class
test_expected<-r$test[,names(r$train)==OUTPUT_FIELD]

#Running the C5.0 decision tree (simple) with 1 trial
print("Create a simple C5.0 decision tree")

tree<-C50::C5.0(x=train_inputs,factor(train_expected),trials=1) # builds the tree
#print(summary(tree))

#Put in test dataset and get out predictions of the decision tree
test_predicted_tree_f<-predict(tree,test_inputs)
test_predicted_tree<-as.numeric(levels(test_predicted_tree_f)[test_predicted_tree_f])

print("Results for C5.0 decision tree on test dataset")
measures_tree<-NcalcConfusion(test_expected,test_predicted_tree,r$test[,27],COST)
N_printMeasures(measures_tree)

#Storing output from the model to a dataset
overall_results<-cbind(data.frame(Algorithm ="Simple C5.0"),measures_tree)

#Running the C5.0 decision tree with BOOST, 20,50,100 were tried and results of 20 and 50 were same hence 20 used
print("Create a simple C5.0 decision tree with boost")

tree2<-C50::C5.0(x=train_inputs,factor(train_expected),trials=BOOST_DT) # builds the tree
#print(summary(tree))

#Put in test dataset and get out predictions of the decision tree
test_predicted_tree2_f<-predict(tree2,test_inputs)
test_predicted_tree2<-as.numeric(levels(test_predicted_tree2_f)[test_predicted_tree2_f])

print("Results for C5.0 decision tree with boost on test dataset")
measures_tree2<-NcalcConfusion(test_expected,test_predicted_tree2,r$test[,27],COST)
N_printMeasures(measures_tree2)

#Storing output from the model to a dataset
c5boost<-as.data.frame(c(Algorithm ="C5.0 with boost 100",measures_tree2 ))
overall_results<-rbind(overall_results,c5boost)

# #Build a linear discriminant classifier on training dataset
# training234<-Nrescaleentireframe(r$train[,1:26])
# str(training234)


#Build a linear discriminant classifier on training dataset
LDAresults<-NLDAperformance(r$train[,1:26], r$test[,1:26], OUTPUT_FIELD)

predicted_decisions_LDA<-NROCthreshold(test_expected,LDAresults$predicted)

print(paste("Linear Discriminant Model"))

measures_LDA<-NcalcConfusion(test_expected,predicted_decisions_LDA,r$test[,27],COST)
N_printMeasures(measures_LDA)

lda_row<-as.data.frame(c(Algorithm ="LDA",measures_LDA ))
overall_results<-rbind(overall_results,lda_row)

# linear discriminant analysis uses Eigenvalues (see text book)
# The (discriminant) coefficients are these scaled eigenvectors
# This function provides $scaling as the contribution of variables into a discriminant
# that is adjusted as variables have different variances and might be measured in different units
# Below gives us just an indication of the "strengh" of each field

strengths<-as.data.frame(LDAresults$model$scaling)

#When a single column dataframe is created R "drops" the frame and creates a vector (weird!)
#So we use the ",drop=FALSE" syntax!
#Order them but strongest positive first

strengths<-strengths[order(strengths$LD1,decreasing = TRUE),,drop=FALSE]
print(strengths)

#plot strengths using the standard R barplot()
par(las=2)
par(mar=c(5,8,4,2))
barplot(strengths$LD1,names.arg = rownames(strengths),horiz = TRUE,cex.names=0.5,space = c(1, 1))

# ************************************************
# RTandom forest of decision trees
FOREST_SIZE<-900 #the algorithm was run multiple times from 300 to 1000 trees. 700 gave the best result
print(paste("Create Random Forest of",FOREST_SIZE,"trees"))
rf<-randomForest::randomForest(train_inputs,factor(train_expected),ntree=FOREST_SIZE ,mtry=sqrt(ncol(train_inputs)))

#Put in test dataset and get out predictions of the decision tree
#This function returns values as R "factors", we convert back to numeric classes
test_predicted_forest<-predict(rf,test_inputs,type="response")
test_predicted_forest<-as.numeric(levels(test_predicted_forest)[test_predicted_forest])

print("Results for Random Forest on test dataset")
measures_rf<-NcalcConfusion(test_expected,test_predicted_forest,r$test[,27],COST)
N_printMeasures(measures_rf)

# ************************************************
# We can get the predicted values out as probabilities and calculate a threshold
# using the ROC

#These are probabilities real values [0,1]
test_predicted_forest<-predict(rf,test_inputs,type="prob")
threshold<-NROCgraph(test_expected,test_predicted_forest[,2])

#We have threshold that maximises TPR and minimises FPR
#Convert this to a "decision" of 1=Churn, 0=NotChurn
test_predicted_forest_class<-ifelse(test_predicted_forest[,2]>=threshold,1,0)

print("Results for Random Forest on test dataset at best")
measures_rf_best<-NcalcConfusion(test_expected,test_predicted_forest_class,r$test[,27],COST)
N_printMeasures(measures_rf_best)

#Storing output from the model to a dataset
rf_best<-as.data.frame(c(Algorithm ="RandomForest 900 trees",measures_rf_best ))
overall_results<-rbind(overall_results,rf_best)

# ************************************************
# SHALLOW BASIC MLP TRAINING
# Using a basic (and slow) backpropagation algorithm

#You can change this to FALSE to not run the basic MLP training
if (TRUE){
  BASICNN_HIDDEN<-10 # experimented with values 1- 10 
  BASICNN_EPOCHS<-25 # experimented with values 1 - 25
  mlp_classifier<-N_LEARN_BasicNeural(r$train[,1:26],OUTPUT_FIELD,BASICNN_HIDDEN,BASICNN_THRES,BASICNN_EPOCHS)
  
  test_predicted_shallowmlp<-N_EVALUATE_BasicNeural(r$test[,1:26],OUTPUT_FIELD, mlp_classifier,DECISION_THR)
  
  #get the best threshold value from the ROC Curve
  threshold_shallowmlp<-NROCgraph(test_expected,test_predicted_shallowmlp)
  
  #We have threshold that maximises TPR and minimises FPR
  test_predicted_class_mlp<-ifelse(test_predicted_shallowmlp>=threshold_shallowmlp,CLASS1,CLASS0 )
  
  print(paste("Results for MLP on test dataset at best ROC threshold",threshold_shallowmlp))
  
  measures_shallowmlp<-NcalcConfusion(test_expected,test_predicted_class_mlp,r$test[,27],COST)
  
  N_printMeasures(measures_shallowmlp)
  
  #Add the results to the overall dataset
  mlp_shallow<-as.data.frame(c(Algorithm ="Shallow MLP NN, Hidden-10 EPOCH-25",measures_shallowmlp))
  overall_results<-rbind(overall_results,mlp_shallow)
  
  readline(prompt="Press [enter] to continue")
}

# # ************************************************
# # DEEP LEARNING EXAMPLE USING H2O library
#
N_DEEP_Initialise()

test_predicted_deep<-N_DEEP_TrainClassifier(r$train[,1:26],r$test[,1:26],OUTPUT_FIELD,DEEP_HIDDEN,DEEP_STOPPING,DEEP_TOLERANCE,DEEP_ACTIVATION,DEEP_REPRODUCABLE)

# test_expected<-r$test[,OUTPUT_FIELD]
threshold_deep<-NROCgraph(test_expected,test_predicted_deep)

#We have threshold that maximises TPR and minimises FPR
test_predicted_class<-ifelse(test_predicted_deep>=threshold_deep,CLASS1,CLASS0)

print(paste("Results for DEEP LEARNING on test dataset at best ROC threshold",threshold_deep))

measures_deep<-NcalcConfusion(test_expected,test_predicted_class,r$test[,27],COST)
N_printMeasures(measures_deep)

#Storing output from the model to a dataset
deep_best<-as.data.frame(c(Algorithm ="Deep NN Hidden-30,10",measures_deep))
overall_results<-rbind(overall_results,deep_best)

print("H2O Deep Learning Complete")
readline(prompt="Press [enter] to continue")

#Output the final result to an HTML table
sort_results<-overall_results[order(-overall_results$roi),]
html_table <- sort_results[,c(1,9:13)]
print(xtable(html_table), type="html", file="overall_results.html")

best_model<-sort_results[1,]
best_model_trim<-sort_results[1,c(1,9:13)]
round_model<-round(best_model_trim[,2:6],2)

#Business case 1, No model and all customers churn on test data
case1_customers_churn<-best_model$TP +best_model$FN
case1_cost_of_acquiring_new_customers<-(case1_customers_churn*COST)*-1

#Business case 2, Best model used and solution implemented
case2_customers_churn<-best_model$FN
case2_revenue_uplift<-best_model$uplift
case2_roi<-round(best_model$roi,2)

#model accuracy
accuracy<-(best_model$TP+best_model$TN)*100/(best_model$TP+best_model$TN+best_model$FP+best_model$FN)

#no of cutomers
totcust<- best_model$TP+best_model$TN+best_model$FP+best_model$FN
  
#Customers churn
barplot(height=c(NoModel = case1_customers_churn,MLModel = case2_customers_churn), main = "Customer Churn", col=c("red","lightgreen"), ylim=range(0,600), ylab="No. of customers")

#Costs
barplot(height=c(NoModel=case1_cost_of_acquiring_new_customers/1000,ML_Uplift=case2_revenue_uplift/1000 ), main = "Customer Churn Impact on Revenue", col=c("red","lightgreen"), ylim=range(-400,200), ylab="Thousands of Dollars $")

#ROI
barplot(height=c(Expenses=(best_model_trim$newCust+best_model_trim$expenses)/1000*-1, Revenue=best_model_trim$revenue/1000), main = "Customer Churn Expenses v/s Revenue using a ML Model", ylab="Thousands of Dollars $", col=c("red","lightgreen"), ylim=range(-200,400))
