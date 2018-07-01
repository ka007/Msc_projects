# *********************************************************************
# Name:CP_functions.R
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
# NPREPROCESSING_removePunctuation()
# INPUT: String - name of field with possible punctuation/spaces
# OUTPUT : String - name of field with punctuation removed
# ************************************************
NPREPROCESSING_removePunctuation<-function(fieldName){
  return(gsub("[[:punct:][:blank:]]+", "", fieldName))
}

# **********************************************************************************************
# Output simple dataset field analysis results as a table in "Viewer"
#
# REQUIRES: formattable
#
# INPUT: Frame - dataset, full dataset used for train/test of the neural network
#              - Each row is one record, each column in named
#              - Values are not scaled or encoded
#
# OUTPUT : none
#
# **********************************************************************************************
NPREPROCESSINGpretty_dataset_table<-function(dataset){
  library(formattable)
  tidyTable<-data.frame(Field=names(dataset),Categorical=FALSE, Symbols=0, Min=0.0, Mean=0.0, Max=0.0,Skew=0.0,stringsAsFactors = FALSE)
  for (i in 1:ncol(dataset)){
    isFieldAfactor<-!is.numeric(dataset[,i])
    tidyTable$Categorical[i]<-isFieldAfactor
    if (isFieldAfactor){
      tidyTable$Symbols[i]<-length(unique(dataset[,i]))  #Number of symbols in catagorical
    } else
    {
      tidyTable$Max[i]<-round(max(dataset[,i]),2)
      tidyTable$Mean[i]<-round(mean(dataset[,i]),2)
      tidyTable$Min[i]<-round(min(dataset[,i]),2)
      tidyTable$Skew[i]<-round(PerformanceAnalytics::skewness(dataset[,i],method="moment"),2)
    }
  }
  
  t<-formattable::formattable(tidyTable,list(Categorical = formatter("span",style = x ~ style(color = ifelse(x,"green", "red")),
                                                                     x ~ icontext(ifelse(x, "ok", "remove"), ifelse(x, "Yes", "No"))),
                                             Symbols = formatter("span",style = x ~ style(color = "black"),x ~ ifelse(x==0,"-",sprintf("%d", x))),
                                             Min = formatter("span",style = x ~ style(color = "black"), ~ ifelse(Categorical,"-",format(Min, nsmall=2, big.mark=","))),
                                             Mean = formatter("span",style = x ~ style(color = "black"),~ ifelse(Categorical,"-",format(Mean, nsmall=2, big.mark=","))),
                                             Max = formatter("span",style = x ~ style(color = "black"), ~ ifelse(Categorical,"-",format(Max, nsmall=2, big.mark=","))),
                                             Skew = formatter("span",style = x ~ style(color = "black"),~ ifelse(Categorical,"-",sprintf("%.2f", Skew)))
  ))
  print(t)
}

# ************************************************
# NPREPROCESSING_splitdataset() : #Randomise and split entire data set
# INPUT: Frame - dataset
#
# OUTPUT : Frame - test dataset
#          Frame - train dataset
# ************************************************
NPREPROCESSING_splitdataset<-function(combinedML){
  
  # **** Create a TRAINING dataset using 70% of the records
  
  combinedML<-combinedML[order(runif(nrow(combinedML))),]
  training_records<-round(nrow(combinedML)*(70/100))
  
  train <- 1:training_records
  test <- -train
  
  training_data <- combinedML[train,]
  testing_data = combinedML[test,]
  
  retList<-list("train"=training_data,
                "test"=testing_data)
  return(retList)
}

# ************************************************
# LINEAR DISCRIMINANT ANALYSIS
# INPUT:      Frame - dataset to create model
#             Fame - dataset to test model
#             String - name of field to predict
# OUTPUT :    model - the LDA object
#             Vector - predicted classes
# ************************************************
# Useslibrary(MASS)
NLDAperformance<-function(training, test, fieldNameOutput){
  
  library(MASS)
  
  inputs<-paste(colnames(training)[which(names(training)!=fieldNameOutput)],collapse = "+")
  output<-paste(fieldNameOutput,"~")
  formular=paste(output,inputs,sep=" ")
  training<-Nrescaleentireframe(training)
  #Build a logistic regression classifier on training dataset
  #Do not seem to be able to pass formula as a string
  LDAmodel<-MASS::lda(as.formula(formular),data=training)
  
  nn<-predict(LDAmodel,test)
  
  retList<-list("model"=LDAmodel,
                "predicted"=as.vector(nn$posterior[,1]))
  
  return(retList)
  
}

# ************************************************
# NROCthreshold()
# INPUT:      Frame - vector of expected classes
#             Frame - vector of predicted classes
# OUTPUT :    Vector - predicted decisions after threshold applied
# ************************************************
# Uses   library(pROC)

NROCthreshold<-function(expected,actual){
  
  library(pROC)
  
  #This is a ROC graph
  
  rr<-roc(expected,actual,plot=FALSE,percent=TRUE)
  
  #Selects the "best" threshold for lowest FPR and highest TPR
  analysis<-coords(rr, x="best",best.method="closest.topleft",
                   ret=c("threshold", "specificity", "sensitivity","accuracy", "tn", "tp", "fn", "fp", "npv","ppv"))
  
  threshold<-analysis[1L]
  print(paste("LDA Threshold=",threshold))
  
  #Convert to decisions
  predicted_decisions<-ifelse(actual>threshold,1.0,0.0)
  
  return(predicted_decisions)
  
}
# ************************************************
# Nrescaleentireframe: rescale each column to range of [0,1]
# INPUT: text - filename
# OUTPUT : Frame - dataset
# ************************************************
#Rescle the entire dataframe to [0.0,1.0]
Nrescaleentireframe<-function(dataset){
  for(field in 1:(ncol(dataset))){
    dataset[,field]<-rescale(dataset[,field],range(0,1))
  }
  return(dataset)
}

# ****************************************************************************
# NPREPROCESSING_redundantFields() : Determine if an entire field is redundant
# based on reoccuring values and collinearity based on cutoff
# INPUT: Frame - dataset
#        float - cutoff - Value above which is determined redundant (0.0-1.0)
# OUTPUT : Frame - dataset with any fields removed
# ****************************************************************************
# Uses LINEAR correlation, so use with care as information will be lost

NPREPROCESSING_redundantFields<-function(dataset,cutoff){
  
  print(paste("Before redundancy check Fields=",ncol(dataset)))
  
  #Remove any fields that have a stdev of zero (i.e. they are all the same)
  xx<-which(apply(dataset, 2, function(x) sd(x, na.rm=TRUE))==0)+1
  
  if (length(xx)>0L)
    dataset<-dataset[,-xx]
  
  #Kendall is more robust for data do not necessarily come from a bivariate normal distribution.
  cr<-cor(dataset, use="everything")
  cr[(which(cr<0))]<-0 #Positive correlation coefficients only
  NPLOT_correlagram(cr)
  
  correlated<-which(abs(cr)>=cutoff,arr.ind = TRUE)
  list_fields_correlated<-correlated[which(correlated[,1]!=correlated[,2]),]
  
  if (length(list_fields_correlated)>0){
    
    print("Following fields are correlated")
    print(list_fields_correlated)
    
    #We have to check if one of these fields is correlated with another as cant remove both!
    v<-vector()
    numc<-nrow(list_fields_correlated)
    for (i in 1:numc){
      if (length(which(list_fields_correlated[i,1]==list_fields_correlated[i:numc,2]))==0) {
        v<-append(v,list_fields_correlated[i,1])
      }
    }
    print("Removing the following fields")
    print(names(dataset)[v])
    
    return(dataset[,-v]) #Remove the first field that is correlated with another
  }
  return(dataset)
}

# ************************************************
# LEARN_BasicNeural() : Train a simple MLP Neural Network classifier
# INPUT: Frame - training_data - scaled [0.0,1.0], fields & rows
#        String - fieldNameOutput - Name of the field that we are training on (i.e.Status)
#        int - hiddenNeurons - Number of hidden layer neurons
#        float - threshold - error value under which the training stops [0.0,1.0]
#        int - trainRep - number of repetitions for the neural network’s training
# OUTPUT : Frame - information, including the learn weights, of the MLP classifier
# ************************************************
# Uses   library(neuralnet)
# https://cran.r-project.org/web/packages/neuralnet/neuralnet.pdf
# Trains a 3 layer, MLP neural network
# Selected a resilient back-propagation with and without weight backtracking algorithm
# Logistic (i.e. sigmoidal) neuron transfer functions so scaled [0.0,1.0]
# Will be slow & no test for overfitting the model

N_LEARN_BasicNeural<-function(training_data,fieldNameOutput,hiddenNeurons,threshold,trainRep){
  
  library(neuralnet)
  
  inputs<-paste(colnames(training_data)[which(names(training_data)!=fieldNameOutput)],collapse = "+")
  output<-paste(fieldNameOutput,"~")
  passtofunction=paste(output,inputs,sep=" ")
  
  print("Training Shallow MLP Neural Network")
  mlp_classifier<-neuralnet(passtofunction,data=training_data,
                            hidden=hiddenNeurons,
                            linear.output=FALSE,
                            act.fct = 'logistic',
                            algorithm="rprop+",
                            threshold = threshold,rep=trainRep)
  
  return(mlp_classifier)
  
}

# ************************************************
# EVALUATE_BasicNeural() : Evaluate a simple MLP Neural Network classifier
#                           Generates predicted classifications from the classifier
# INPUT: Frame - testing_data - scaled [0.0,1.0], fields & rows
#        String - fieldNameOutput - Name of the field that we are training on (i.e.Status)
#        Frame - mlp_classifier - structure including the learn weights, of the MLP classifier
#        Float - cutoff - value over which is determined to be a "yes" (1) decision
# OUTPUT :None
# ************************************************
# Uses   library(neuralnet)
# https://cran.r-project.org/web/packages/neuralnet/neuralnet.pdf

N_EVALUATE_BasicNeural<-function(testing_data,fieldNameOutput, mlp_classifier,cutoff){
  
  positionOutput<-which(names(testing_data)==fieldNameOutput)
  
  res<-compute(mlp_classifier,testing_data[,-positionOutput])$net.result
  
  return(as.vector(res))
}

# ************************************************
# NROCgraph() : This is a ROC graph
# INPUT:        Frame - dataset to create model
#               Fame - dataset to test model
# OUTPUT :      Float - calculated thresholkd from ROC
# ************************************************
NROCgraph<-function(expected,predicted){
  
  library(pROC)
  
  rr<-roc(expected,predicted,plot=FALSE,percent=TRUE,partial.auc=c(100, 75), partial.auc.correct=TRUE,partial.auc.focus="sens",uc.polygon=TRUE,
          max.auc.polygon=TRUE, grid=TRUE,print.auc=TRUE, show.thres=TRUE,add=FALSE,xlim=c(1,0),main="ROC Forest Model")
  
  plot(rr,xlim=c(100,0),xaxs="i")
  
  #Selects the "best" threshold for lowest FPR and highest TPR
  analysis<-coords(rr, x="best",best.method="closest.topleft",
                   ret=c("threshold", "specificity", "sensitivity","accuracy", "tn", "tp", "fn", "fp", "npv","ppv"))
  
  threshold<-analysis[1L]
  specificity<-analysis[2L]
  sensitivity<-analysis[3L] #same as TPR
  fpr<-round(100.0-specificity,digits=2L)
  
  #Add crosshairs to the graph
  abline(h=sensitivity,col="red",lty=3,lwd=2)
  abline(v=specificity,col="red",lty=3,lwd=2)
  
  #Annote with text
  text(x=specificity,y=sensitivity, adj = c(-0.2,2),cex=1,
       col="red",
       paste("Threshold: ",round(threshold,digits=4L),
             " TPR: ",round(sensitivity,digits=2L),
             "% FPR: ",fpr,"%",sep=""))
  
  return(threshold)
}

# ************************************************
# N_DEEP_Initialise()
# Initialise the H2O server
# INPUT: none
# OUTPUT : none
# ************************************************
# Connect to the H2O system

N_DEEP_Initialise<-function(){
  
  library(h2o)
  
  print("Initialise the H2O server")
  #Initialise the external h20 deep learning local server if needed
  #Updated- set nthreads to -1 to use maximum so fast, but set to 1 to get reproducable results
  
  h2o.init(max_mem_size = "5g",nthreads = 1)
  #h2o.no_progress()
}

# ************************************************
# N_DEEP_TrainClassifier()
# NEURAL NETWORK : DEEP LEARNING CLASSIFIER TRAIN
# INPUT:  Frame - train - scaled [0.0,1.0], fields & rows
#         Frame - test - scaled [0.0,1.0], fields & rows
#         String - fieldNameOutput - Name of the field that we are training on (i.e.Status)
#         Int Vector -  hidden - Number of hidden layer neurons for each layer
#         int - stopping_rounds - Number of times no improvement before stop
#         float - stopping_tolerance - Error threshold
#         String - activation - Name of activation function
#         Bool - checkReproducible - true if debug test that training is reproducable each run
# OUTPUT: Float Vector - probabilities of class 1
# ************************************************
N_DEEP_TrainClassifier<- function(train,test,fieldNameOutput,hidden,stopping_rounds,stopping_tolerance,activation, checkReproducible){
  
  positionOutput<-which(names(test)==fieldNameOutput)
  
  #Creates the h2o training dataset
  train[fieldNameOutput] <- lapply(train[fieldNameOutput] , factor) #Output class has to be a R "factor"
  train_h2o <- as.h2o(train, destination_frame = "traindata")
  
  #Creates the h2o test dataset
  test[fieldNameOutput] <- lapply(test[fieldNameOutput] , factor) #Output class has to be a R "factor"
  test_h2o <- as.h2o(test, destination_frame = "testdata")
  
  #This lists all the input field names ignoring the fieldNameOutput
  predictors <- setdiff(names(train_h2o), fieldNameOutput)
  
  #Deep training neural network.  NOTE: Should use cross-validation with a test dataset
  #Updated 13/5/17 - set reproducible = TRUE so that the same random numbers are used to initalise
  
  deep<-h2o.deeplearning(x=predictors,y=fieldNameOutput,training_frame = train_h2o,
                         epochs=200,
                         hidden=hidden,
                         adaptive_rate=TRUE,
                         stopping_rounds=stopping_rounds,
                         stopping_tolerance=stopping_tolerance,
                         fast_mode=FALSE,
                         activation=activation,
                         seed=1234,
                         reproducible = TRUE)
  
  #TEST IF THE MODEL IS REPRODUCABLE OTHERWISE ABORT
  if (checkReproducible==TRUE){
    
    #calculate AUC for first model
    auc1<-h2o.auc(deep)
    
    deep<-h2o.deeplearning(x=predictors,y=fieldNameOutput,training_frame = train_h2o,
                           epochs=200,
                           hidden=hidden,
                           adaptive_rate=TRUE,
                           stopping_rounds=stopping_rounds,
                           stopping_tolerance=stopping_tolerance,
                           fast_mode=FALSE,
                           activation=activation,
                           seed=1234,
                           reproducible = TRUE)
    
    auc2<-h2o.auc(deep)
    
    if (auc1!=auc2)
      stop("DEEP LEARN IS NOT REPRODUCABLE")
  }
  
  # ************************************************
  # TELL ME SOMETHING INTERESTING...
  summary(deep)
  plot(deep)  # plots the scoring history
  
  # variable importance
  var_imp = h2o.varimp(deep)
  
  for (rows in 1:nrow(var_imp)){
    print(paste(var_imp$variable[rows],round(var_imp$scaled_importance[rows],digits=3),round(var_imp$percentage[rows],digits=2)))
  }
  
  # ************************************************
  
  pred <- h2o.predict(deep, test_h2o)
  
  res<-as.vector(pred[,3])  #Returns the probabilities of class 1
  return(res)
}


# ************************************************
# Various measures : Calculate various confusion matrix measures
# INPUT: TP - int - True Positive records
#        FP - int - False Positive records
#        TN - int - True Negative records
#        FN - int - False Negative records
# OUTPUT : float - calculated results
# ************************************************

# NcalcAccuracy<-function(TP,FP,TN,FN){return(100.0*((TP+TN)/(TP+FP+FN+TN)))}
# NcalcPgood<-function(TP,FP,TN,FN){return(100.0*(TP/(TP+FP)))}
# NcalcPbad<-function(TP,FP,TN,FN){return(100.0*(TN/(FN+TN)))}
NcalcFPR<-function(TP,FP,TN,FN){return(100.0*(FP/(FP+TN)))}
NcalcFNR<-function(TP,FP,TN,FN){return(100.0*(FN/(TP+FN)))}
NcalcMCC<-function(TP,FP,TN,FN){return( ((TP*TN)-(FP*FN))/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))}


# ************************************************
# Nrmse() : Calculate the RMSE statistic
# INPUT: actual_y vector of real numbers indicating the known class
#        y_predicted vector of real numbers indicating the predicted class
# OUTPUT : Frame - dataset
# ************************************************
Nrmse<-function(actual_y,y_predicted){
  
  return(sqrt(mean((actual_y-y_predicted)^2)))
}

# ************************************************
# NcalcMeasures() : Calculate measures from values in confusion matrix
# INPUT: int TP, FN, TN, FN
#
# OUTPUT: A list with the following entries:
#        accuracy - float - accuracy measure
#        pgood - float - precision for "good" (values are 1) measure
#        pbad - float - precision for "bad" (values are 1) measure
#        FPR - float - FPR measure
#        FNR - float - FNR measure
#        MCC - float - Matthew's Correlation Coeficient
# ************************************************

NcalcMeasures<-function(TP,FN,TN,FP) {
  retList<-list(  "TP"=TP,
                  "FN"=FN,
                  "TN"=TN,
                  "FP"=FP,
                  # "accuracy"=NcalcAccuracy(TP,FP,TN,FN),
                  # "pgood"=NcalcPgood(TP,FP,TN,FN),
                  # "pbad"=NcalcPbad(TP,FP,TN,FN),
                  "FPR"=NcalcFPR(TP,FP,TN,FN),
                  "FNR"=NcalcFNR(TP,FP,TN,FN),
                  "MCC"=NcalcMCC(TP,FP,TN,FN)
  )
  return(retList)
}

# ************************************************
# NcalcConfusion() : Calculate a confusion matrix for 2-class classifier
# INPUT: vector - expected - {0,1}, Expected outcome from each row (labels)
#        vector - predicted - {0,1}, Predicted outcome from each row (labels)
#        vector - actualp - [0.00], actual monthly charges
#        float - cost - cost of acquiring a new customer
#
# OUTPUT: A list with the following entries:
#        TP - int - True Positive records
#        FP - int - False Positive records
#        TN - int - True Negative records
#        FN - int - False Negative records
#        expenses - Total expenses which would be incurred on marketing promotions
#        revenue - Total revenue impact due to churn
#        newCust - Total Costs incurred towards acquiring a newCust, 1 for every FN 
#        roi - Return on investment (savings on revenue made)/expenses
#        uplift<- Money saved for the company by using the model on test data
#        accuracy - float - accuracy measure
#        pgood - float - precision for "good" (values are 1) measure
#        pbad - float - precision for "bad" (values are 1) measure
#        FPR - float - FPR measure
#        TPR - float - FPR measure
#        MCC - float - Matthew's Correlation Coeficient
# ************************************************

NcalcConfusion<-function(expected,predicted,actualp,cost){
  
  #Marked in the dataset, set as "1" for the class we are looking for
  
  TP<-0 #A fraud transaction was expected and was correctly classified by the decision system.
  FN<-0 #A fraud transaction was expected but was wrongly classified as genuine.
  TN<-0 #A genuine transaction was expected and was correctly classified by the decision system.
  FP<-0 #A genuine transaction was expected and was wrongly classified as fraud.
  expenses<-0 #Total expenses which would be incurred on marketing promotions
  revenue<-0  #Total revenue impact due to churn
  newCust<-0  #Total Costs incurred towards acquiring a newCust, 1 for every FN
  roi<-0      #Return on investment
  uplift<-0   #Total savings made using the model minus the expenses incurred
  
  
  for (x in 1:length(predicted)){
    fire<-predicted[x]
    marked<-expected[x]
    charges12<-actualp[x]*12 #actualp is the monthly cost for customerx, 
    
    #In the case of a POSITIVE
    if (fire==TRUE){
      #A fraud transaction was expected and was correctly classified by the rules
      #TRUE POSITIVE
      if (marked==1.0){
        TP<-TP+1
        expenses<-expenses +(0.1*charges12)
        revenue<-revenue + charges12
      }
      else
      {
        #A genuine transaction was expected and was wrongly classified as fraud by the rules
        #FALSE POSITIVE
        FP<-FP+1
        expenses<-expenses + (0.1*charges12)
      }
    }
    else {
      #A genuine transaction was expected and was correctly classified by the rules
      #TRUE NEGATIVE
      if (marked==0.0){
        TN<-TN+1
      }
      else
      {
        #A fraud transaction was expected but was wrongly classified as genuine by the rules
        #FALSE NEGATIVE
        FN<-FN+1
        revenue<-revenue - charges12
        newCust<-newCust + cost
      }
    }
  }
  
  RMSE<-round(Nrmse(expected,predicted),digits=2)
  
  # return on investment where revenue is savings on revenue made due to the model
  roi<-(revenue)/(expenses+newCust) 
  
  # Uplift Total of revenue - expenses
  uplift<- revenue -(expenses+newCust)
  measure<-NcalcMeasures(TP,FN, TN, FP )
  
  retList<-list(  "TP"=TP,
                  "FN"=FN,
                  "TN"=TN,
                  "FP"=FP,
                  # "accuracy"=measure$accuracy,
                  # "pgood"=measure$pgood,
                  # "pbad"=measure$pbad,
                  "FPR"=measure$FPR,
                  "FNR"=measure$FNR,
                  "MCC"=measure$MCC,
                  "expenses"=expenses,
                  "revenue"=revenue,
                  "newCust"=newCust,
                  "roi"=roi,
                  "uplift"=uplift
  )
  return(retList)
}

# ************************************************
# N_printMeasures()
# Prints measures to the console
#
# INPUT:    List of results from NcalcConfusion()
# OUTPUT :  NONE
# ************************************************
N_printMeasures<-function(results){
  print(paste("TP ",round(results$TP,2)))
  print(paste("FN ",round(results$FN,2)))
  print(paste("TN ",round(results$TN,2)))
  print(paste("FP ",round(results$FP,2)))
  print(paste("FPR ",round(results$FPR,2)))
  print(paste("FNR ",round(results$FNR,2)))
  print(paste("MCC ",round(results$MCC,4)))
  print(paste("Expense $",round(results$expenses,2)))
  print(paste("Revenue $",round(results$revenue,2)))
  print(paste("New Customer Cost $ ",round(results$newCust,2)))
  print(paste("Uplift $ ",round(results$uplift,2)))
  print(paste("ROI ",round(results$roi,2)))
}

# ************************************************
# NPLOT_correlagram() : Plots PLOT_correlagram
# INPUT: Frame - cr - n x n frame of correlation coefficients for all fields
# OUTPUT : None
# ************************************************
NPLOT_correlagram<-function(cr){
  
  library(corrplot)
  #Defines the colour range
  col<-colorRampPalette(c("green", "red"))
  
  #To fir on screen, convert field names to a numeric
  rownames(cr)<-1:length(rownames(cr))
  colnames(cr)<-rownames(cr)
  
  corrplot(cr,method="square",order="FPC",cl.ratio=0.2, cl.align="r",tl.cex = 0.6,cl.cex = 0.6,cl.lim = c(0, 1),mar=c(1,1,1,1))
}
