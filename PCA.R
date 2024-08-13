setwd("C:/Users/User/Desktop/R")


# Read data from the file
data <- read.table("T8-5.DAT") 

colnames(data) <- c("Total population(thousands)", "Professional degree(percent)", "Employed age over 16(percent)", "Government employment(percent)", "Median value($)")
head(data)
dim(data)
str(data)
colSums(is.na(data))
data_normalized<-scale(data)
head(data_normalized)
installed.packages("earth")
install.packages("caTools")

install.packages(c("psych"))installed.packages()
install.packages("lsr")
install.packages(c("psych","GPArotation","sem","matrixcalc"))
install.packages("corrplot")
install.packages("corrr")
install.packages("ggcorrplot")
library(ggcorrplot)
install.packages("FactoMineR")
install.packages("Factorextra")
library("FactoMineR")
Compute the correlation matrix and hence visualize

#install.packages("corrplot")

corr_matrix <- cor(data_normalized)
corr_matrix <- cor(data_normalized)
corrplot(corr_matrix)
ggcorrplot(corr_matrix)
##performing PCA and summarize the results
data.pca <- princomp(corr_matrix)

summary(data.pca)
####PCA loadings:provides the eigenvalues and vectors
data.pca$loadings
####How many PCs shoud be retained?
fviz_eig(data.pca, addlabels = TRUE)



# Principal component analysis
#' # +++++++++++++++++++++++++++++
#' data(decathlon2)
#' decathlon2.active <- decathlon2[1:23, 1:10]
#' res.pca <- prcomp(decathlon2.active,  scale = TRUE)
#' fviz(res.pca, "ind") # Individuals plot
#' fviz(res.pca, "var") # Variables plot
#' 
#' # Correspondence Analysis
#' # ++++++++++++++++++++++++++
#' # Install and load FactoMineR to compute CA
#' # install.packages("FactoMineR")
#' library("FactoMineR")
#' data("housetasks")
#' res.ca <- CA(housetasks, graph = FALSE)
#' fviz(res.ca, "row") # Rows plot
#' fviz(res.ca, "col") # Columns plot
#' nstall.packages("FactoMineR")

Load FactoMineR in your R session by writing the following line code:
  library(FactoMineR)

Install its companion packages
You can install several companion packages for FactoMineR: Factoshiny to have a graphical interface that draws graph interactively, missMDA to handle missing values, and FactoInvestigate to obtain automatic description of your analyses.

Download the packages you want:
  install.packages(c("Factoshiny","missMDA","FactoInvestigate"))

Load the packages in your R session when you want to use them by writing the following lines of code:
  library(Factoshiny)
library(missMDA)
library(FactoInvestigate)
#' 
#' # Multiple Correspondence Analysis
#' # +++++++++++++++++++++++++++++++++
#' library(FactoMineR)
#' data(poison)
#' res.mca <- MCA(poison, quanti.sup = 1:2, 
#'               quali.sup = 3:4, graph=FALSE)
#'               
#' fviz(res.mca, "ind") # Individuals plot
#' fviz(res.mca, "var") # Variables plot


# Graph of the variables
Biplot of the variables with respect to the principal components
####visualize the similarities and dissimilarities between the samples
####shows the impact of each attribute on each of the principal components
fviz_pca_var(data.pca, col.var = "black")
corr_matrix<-cor(data_normalized)
corrplot(corr_matrix)

data.pca< princomp(corr_matrix)
summary(data.pca)
data.pca$loadings

' # Default plot
#' fviz_eig(res.pca, addlabels = TRUE, ylim = c(0, 85))
#'   
#' # Scree plot - Eigenvalues
#' fviz_eig(res.pca, choice = "eigenvalue", addlabels=TRUE)
#' 
#' # Use only bar  or line plot: geom = "bar" or geom = "line"
#' fviz_eig(res.pca, geom="line")
-
  install.packages("FactoMineR")

Load FactoMineR in your R session by writing the following line code:
  library(FactoMineR)

Install its companion packages
You can install several companion packages for FactoMineR: Factoshiny to have a graphical interface that draws graph interactively, missMDA to handle missing values, and FactoInvestigate to obtain automatic description of your analyses.

Download the packages you want:
  install.packages(c("Factoshiny","missMDA","FactoInvestigate"))

Load the packages in your R session when you want to use them by writing the following lines of code:
  library(Factoshiny)
library(missMDA)
library(FactoInvestigate)
install.packages("devtools")
install.packages("FactoMineR")
install.packages("factoextra")
corr_matrix <- cor(data_normalized)
ggcorrplot(corr_matrix)
data.pca <- princomp(corr_matrix)
summary(data.pca)
data.pca$loadings[, 1:2]
fviz_eig(data.pca, addlabels = TRUE)

fviz_eig(data.pca, addlabels = TRUE)
Contribution of each variable
###low value: the variable is not perfectly represented by that component
###high value: means a good representation of the variable on that component.
fviz_eig(data.pca, addlabels = TRUE)
"var", axes = 1:2)
fviz_eig(data.pca, addlabels = TRUE)
# Graph of the variables
fviz_cos2(data.pca, choice = 
fviz_pca_var(data.pca, col.var = "black")
Contribution of each variable
###low value: the variable is not perfectly represented by that component
###high value: means a good representation of the variable on that component.
fviz_cos2(data.pca, choice = "var", axes = 1:2)

hjhjBiplot combined with cos2
####High cos2 attributes are colored in green
####Mid cos2 attributes have an orange color
####Low cos2 attributes have a black color

fviz_pca_var(data.pca, col.var = "cos2",
             gradient.cols = c("black", "orange", "green"),
             repel = TRUE)
corr_matrix <- cor(data_normalized)
ggcorrplot(corr_matrix)
fviz_eig()
library(quarto)
quarto_render("document.qmd") # all formats
quarto_render("document.qmd", output_format = "pdf")
install.packages("rmarkdown")
install.packages("rmarkdown")
data<-read.table("T8-4.DAT")
colnames(data)<-c("JP Morgan","Citibank","Wells Fargo",
                  "Royal Dutch Sheel","Exxon Mobil")
head(data)

###Check the data structure and missing values
str(data)
colSums(is.na(data))
#KMO(r=data)
#####Compute the correlation matrix and hence visualize
#install.packages("corrplot")
library(corrplot)
R<- cor(data)
corrplot(R)
install.packages("nFactors")

library(nFactors)

## Loading required package: lattice
ev <- eigen(cor(data)) # get eigenvalues
####Parallel analysis of a correlation or covariance Matrix
####Finds the distribution of the eigenvalues of correlation or a covariance matrices.
ap <- parallel(subject=nrow(data),var=ncol(data),
               rep=100,cent=.05)
nS <- nScree(x=ev$values, aparallel=ap$eigen$qevpea)
plotnScree(nS)

## Loading required package: lattice
install.packages("lattice")
## 
## Attaching package: 'nFactors'
## The following object is masked from 'package:lattice':
## 
##     parallel
cor(data)
ev = eigen(cor(data))
ev
## eigen() decomposition
## $values
ev <- eigen(cor(data)) # get eigenvalues

install.packages("nFactors")
library(nFactors)
install.packages("lattice")
install.packages("Parallel")
# Load nFactors without attaching it
library(nFactors)

# Use the parallel function from the lattice package
lattice::parallel(...)

ev <- eigen(cor(data)) # get eigenvalues
####Parallel analysis of a correlation or covariance Matrix
####Finds the distribution of the eigenvalues of correlation or a covariance matrices.
ap <- parallel(subject=nrow(data),var=ncol(data),
               rep=100,cent=.05)
nS <- nScree(x=ev$values, aparallel=ap$eigen$qevpea)
plotnScree(nS)
####Parallel analysis of a correlation or covariance Matrix
install.packages("parallelly")
library(nFactors)
ap = parallel(subject = nrow(X),
              var = ncol(X), rep=100,cent=0.05)
# parallel: This makes a correlation matrices from null dist
ns = nScree(ev$values,ap$eigen$qevpea)
plotnScree(ns)
####Finds the distribution of the eigenvalues of correlation or a covariance matrices.
ap <- parallelplot(subject=nrow(data),var=ncol(data), rep=100,cent=.05)
nS <- nScree(x=ev$values, aparallel=ap$eigen$qevpea)
plotnScree(nS)
fa <- factanal(data, factors = 2,covmat = R)
fa
fa$method
## [1] "mle"
fa$loadings
1-fa$uniquenesses
apply(fa$loadings^2,1,sum)###another way
fa$uniquenesses
L<-fa$loadings#factor loadings from previous analysis
psi<-diag(fa$uniquenesses)
R<-cor(data)
R_es<-L%*%t(L)+psi
R_es
res<-round(R-R_es,6)
res
fa
plot(fa$loadings[, 1],fa$loadings[, 2],xlab = "Factor 1",ylab = "Factor 2",  ylim = c(-1, 1), xlim = c(-1, 1),main = "")
abline(h = 0, v = 0)
text(fa$loadings[, 1] - 0.08,fa$loadings[, 2] + 0.08,colnames(data),
     col = "red")
pca<-PCA(data)
repca<-princomp(data)
summary(repca)
plot(repca)
fviz_eig(repca, addlabels = TRUE)
fviz_pca_var(repca, col.var = "black")
fviz_cos2(repca, choice = "var", axes = 1:2)
data<-read.table("T12-4.DAT")
colnames(data)<-c("X1","X2","X3","X4","X5","X6","X7","X8","Company")
str(data)
colSums(is.na(data))
data_normalized<-scale(data[,1:8])
R<-cor(data_normalized)
corrplot(R,method=c("circle"),type="lower",
         title="Public Utility Data: Correlation Matrix", addCoef.col= "red",cex.main=0.7)
corrplot(R,method=c("circle"),type="lower",
         title="Public Utility Data: Correlation Matrix", addCoef.col= "red", cex.main=0.7, order="hclust")
ndata<-t(data_normalized)
dmat<-dist(ndata,method= "euclidean")
dmat
############Clustering in heatmap
heatmap(as.matrix(dmat),col=heat.colors(10)
      ###another way
 library(gplots)
 heatmap.2(as.matrix(dmat))
  ################Hierarchical Clustering
 ###Agglomerative
 claS<-hclust(dmat,method="single")
 plot(claS,xlab="Variables",ylab="Distance",
             main="Cluster Dendogram: Public Utility Data",cex.main=1)
 install.packages("mclust")
library(mclust)
groups <-cutree(claS, k=2) # cut tree into 5 clusters
# draw dendogramwith red borders around the 5 clusters
rect.hclust(claS, k=3, border="red")

claC<-hclust(dmat,method="complete")
plot(claC,xlab="Variables",ylab="Distance",
     main="Cluster Dendogram: Public Utility Data",cex.main=1)
claA<-hclust(dmat,method="average")
claA
plot(claA,xlab="Variables",ylab="Distance",
     main="Cluster Dendogram: Public Utility Data",cex.main=1)
claW<-hclust(dmat,method="ward.D2")
claW
plot(claW,xlab="Variables",ylab="Distance",
     main="Cluster Dendogram: Public Utility Data",cex.main=1)
Cluster Analysis
############data
data<-iris
dim(data);str(data)
install.packages("ClusterR")
;library(ClusterR)
install.packages("cluster")
;library(cluster)
#Packages
install.packages("ClusterR");library(ClusterR)
install.packages("cluster");library(cluster)
library(factoextra)
# Removing label of Species from original dataset
ndata<-data[, -5]
##########optimal number of clusters
###using Total within-cluster sum of squares
fviz_nbclust(ndata,kmeans,method="wss")
Fitting K-Means clustering Model to training dataset
km1<-kmeans(ndata, centers = 3, nstart= 20)
km1
# Cluster identification foreachobservation
km1$cluster
# Visualize the clusters
fviz_cluster(km1, data = ndata)
# Confusion Matrix
cm<-table(data$Species, km1$cluster)
cm
# Model Evaluation and visualization
plot(ndata[c("Sepal.Length", "Sepal.Width")])
plot(ndata[c("Sepal.Length", "Sepal.Width")],col = km1$cluster, main = "K-means with 3 clusters")
## Plotingcluster centers
km1$centers
km1$centers[, c("Sepal.Length", "Sepal.Width")]
points(km1$centers[, c("Sepal.Length", "Sepal.Width")],
       col = 1:3, pch= 8, cex= 3)
## Visualizing clusters
ct<-km1$cluster
clusplot(ndata[, c("Sepal.Length", "Sepal.Width")],
         ct,lines= 0,color = TRUE,labels= 2,xlab = 'Sepal.Length',ylab= 'Sepal.Width',main="")


View
# Install and load necessary packages
install.packages("tidyverse")
install.packages("caret")
install.packages("randomForest")
install.packages("rpart")

library(tidyverse)
library(caret)
library(randomForest)
library(rpart)
setwd("C:/Users/User/Desktop/R")
# Read data from the file
housing_data <- read_csv("Housing - Housing.csv")



# 1. Data Loading and Exploration
# a. Load the housing prices dataset


# b. Explore the dataset's structure, dimensions, and summary statistics
str(housing_data)
summary(housing_data)




# b. Use appropriate techniques to ensure a random and representative split

# 3. Data Pre-processing
# a. Handle missing values, if any
# Assuming no missing values for simplicity

# 1. Data Loading and Exploration
# a. Load the housing prices dataset
colSums(is.na(housing_data)
summary(housing_data)
# 2. Data Splitting
# a. Split the dataset into a training set (80%) and a test set (20%)
set.seed(123) # Set seed for reproducibility
split_index <- (housing_data$price, p = 0.8, list = FALSE)
train_data <- housing_data[split_index, ]
test_data <- housing_data[-split_index, ]

# 2. Data Splitting
# a. Split the dataset into a training set (80%) and a test set (20%)
set.seed(123) # Set seed for reproducibility
split_index <- (housing_data$price, p = 0.8, list = FALSE)
train_data <- housing_data[split_index, ]
test_data <- housing_data[-split_index, ]

# b. Use appropriate techniques to ensure a random and representative split

# 3. Data Pre-processing
# a. Handle missing values, if any
# Assuming no missing values for simplicity

# b. Perform feature scaling or normalization
# Normalize numeric features
train_data_scaled <- scale(train_data[, c("area", "bedrooms", "bathrooms")])
train_data[, c("area", "bedrooms", "bathrooms")] <- train_data_scaled

test_data_scaled <- scale(test_data[, c("area", "bedrooms", "bathrooms")])
test_data[, c("area", "bedrooms", "bathrooms")] <- test_data_scaled

# c. Encode categorical variables
train_data <- dummyVars(" ~ .", data = train_data) %>% predict(train_data)
test_data <- dummyVars(" ~ .", data = test_data) %>% predict(test_data)

# 4. Model Selection
# a. Choose at least three regression models
# b. Train each model using the training set
lm_model <- lm(price ~ ., data = train_data)
rf_model <- randomForest(price ~ ., data = train_data)
# Add more models as needed

# 5. Model Evaluation
# a. Predict housing prices using the test set
lm_pred <- predict(lm_model, newdata = test_data)
rf_pred <- predict(rf_model, newdata = test_data)
# Add predictions for other models as needed

# b. Evaluate the models using suitable regression metrics
lm_metrics <- postResample(lm_pred, test_data$price)
rf_metrics <- postResample(rf_pred, test_data$price)
# Add metrics for other models as needed

# c. Compare the performance of the models
comparison_table <- data.frame(
  Model = c("Linear Regression", "Random Forest"),
  RMSE = c(sqrt(mean(lm_metrics^2)), sqrt(mean(rf_metrics^2)))
)
# Add performance metrics for other models as needed

# 6. Hyperparameter Tuning
# a. Select the best-performing model
# b. Perform hyperparameter tuning using techniques like GridSearchCV or RandomizedSearchCV
# Hyperparameter tuning for Random Forest example
tune_grid <- expand.grid(mtry = c(2, 4, 6, 8))
rf_tuned_model <- train(price ~ ., data = train_data, method = "rf", tuneGrid = tune_grid)

# 7. Visualization
# a. Visualize the predicted prices against the actual prices using scatter plots
plot(test_data$price, lm_pred, main = "Linear Regression: Actual vs. Predicted Prices", col = "blue", pch = 16)
abline(0, 1, col = "red")

# b. Create a learning curve to analyze model performance with varying amounts of training data
learning_curve <- data.frame(
  Training_Size = seq(0.1, 0.9, by = 0.1),
  RMSE = numeric(length = 9)
)

for (i in 1:9) {
  subset_size <- round(nrow(train_data) * learning_curve$Training_Size[i])
  subset <- train_data[1:subset_size, ]
  model <- lm(price ~ ., data = subset)
  predictions <- predict(model, newdata = test_data)
  learning_curve$RMSE[i] <- sqrt(mean((test_data$price - predictions)^2))
}

# 8. Conclusion
# a. Summarize the findings and insights gained from the analysis
# b. Reflect on the challenges encountered and potential improvements
```

This is a basic outline, and you might need to adapt it based on the specifics of your dataset and requirements. Additionally, consider adding more models and tuning parameters based on your analysis

library(caret)
library(tidyverse)
library(MASS)
#############################
data<-iris
# Split the data into training (80%) and test set (20%)
set.seed(123)
t.samples<-data$Species%>%
createDataPartition(p = 0.8, list = FALSE)
train.data<-data[t.samples, ]
test.data<-iris[-t.samples, ]
# Estimate pre-processing parameters
pre.param<-train.data%>%preProcess(method = c("center", "scale"))
# Transform the data using the estimated parameters
train.trans<-pre.param%>% predict(train.data)
test.trans<-pre.param%>% predict(test.data)
# Fit the model
model <-lda(Species~., data = train.trans)
plot(model)
lda.data<-cbind(train.trans, predict(model)$x)
ggplot(lda.data, aes(LD1, LD2)) +
  geom_point(aes(color = Species))
# Make predictions
predictions <-model %>% predict(test.trans)
####model accuracy
mean(predictions$class==test.trans$Species)
table(predictions$class==test.trans$Species)
cm<-confusionMatrix(test.trans$Species,predictions$class)
cm
install.packages.("caret")
install.packages("caret", dependencies = c("Depends", "Suggests"))
library(caret)
library(MASS)
library(ggplot2)
#scale each predictor variable (i.e. first 4 columns)
iris[1:4] <- scale(iris[1:4]
  #find mean of each predictor variable
 apply(iris[1:4], 2, mean) 
 #find standard deviation of each predictor variable
 #make this example reproducible
 set.seed(1)
 
 
 
 #Use 70% of dataset as training set and remaining 30% as testing set
 sample <- sample(c(TRUE, FALSE), nrow(iris), replace=TRUE, prob=c(0.7,0.3))
 train <- iris[sample, ]
 test <- iris[!sample, ] 
 #fit LDA model
 model <- lda(Species~., data=train)
 
 #view model output
 model
 Call:
 lda(Species ~ ., data = train)
 
 #use LDA model to make predictions on test data
 predicted <- predict(model, test)
 
 names(predicted)
 
 [1] "class"     "posterior" "x" 
 
 #view predicted class for first six observations in test set
 head(predicted$class)
 
 [1] setosa setosa setosa setosa setosa setosa
 Levels: setosa versicolor virginica
 
 #view posterior probabilities for first six observations in test set
 head(predicted$posterior)
 #view linear discriminants for first six observations in test set
 head(predicted$x)
 #find accuracy of model
 mean(predicted$class==test$Species)
 #define data to plot
 lda_plot <- cbind(train, predict(model)$x)
 
 #create plot
 ggplot(lda_plot, aes(LD1, LD2)) +
   geom_point(aes(color = Species))
 
 ####model accuracy
 # Make predictions
 predictions <-model %>% predict(test.trans)
 ####model accuracy
 mean(predictions$class==test.trans$Species)
 table(predictions$class==test.trans$Species)
 cm<-confusionMatrix(test.trans$Species,predictions$class)
 cm
 mean(predicted$class==test.trans$Species)
 table(predicted$class==test.trans$Species)
 cm<-confusionMatrix(test.trans$Species,predictions$class)
 cm