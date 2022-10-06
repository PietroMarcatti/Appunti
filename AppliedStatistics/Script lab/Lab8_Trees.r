library(splines)
library(mgcv)
library(xtable)
library(DAAG)
library(rpart)
library(RWeka)
library(randomForest)
library(rpart)
library(ISLR)
library(gbm)
library(crossval)

#######################################
# Introduction to tree-based methods
#######################################
##############################
# Example: the spam data set
##############################
par(mfrow=c(1,1))
spam.rpart <- rpart(yesno~crl.tot+dollar+bang+money+n000+make, method="class", data=spam7)
plot(spam.rpart)
text(spam.rpart, cex=1.3)

#######################
# Regression trees
#######################
# a simple example with only one esplicative variable
plot(Mileage~Weight, data = car.test.frame, xlab = 'Weight', ylab = 'Mileage', cex=0.8, pch = 16)
with(car.test.frame, lines(lowess(Mileage~Weight, f=.7), lwd=2, col='blue'))
mod.lm <- lm(log(Mileage)~Weight,car.test.frame)
wseq <- seq(min(car.test.frame$Weight), max(car.test.frame$Weight), l=30)
lines(wseq, exp(predict(mod.lm, data.frame(Weight=wseq))),col=2, lwd=2)
# the regression tree
car.tree <-  rpart(Mileage ~ Weight, data=car.test.frame, method="anova")

car1 <- rpart(Mileage ~ Weight, data=car.test.frame, method="anova",
control=list(cp=0.5))
# cp is the complexity parameter.
# For instance, with anova splitting, 
# this means that the overall R-squared must 
# increase by cp at each step. 
y1 <- predict(car1)
car2 <- rpart(Mileage ~ Weight, data=car.test.frame, method="anova",
control=list(cp=0.1))
y2 <- predict(car2)
y3 <- predict(car.tree)
par(mfrow=c(2,2))
with(car.test.frame, plot(Mileage~Weight, pch=16))
abline(h=mean(car.test.frame$Mileage), col=1, lwd=2)
title("No split")
with(car.test.frame, plot(Mileage~Weight, pch=16))
abline(v=2567.5, col=4, lwd=1.5) 
segments(1500, max(y1),  2567.5, max(y1), col=1, lwd=2)
segments(2567.5, min(y1), 4000, min(y1), col=1, lwd=2)
title("First split")
with(car.test.frame, plot(Mileage~Weight, pch=16))
abline(v=2567.5, col=4, lwd=1.5) 
abline(v=3087.5, col=4, lwd=1.5) 
#points(car.test.frame$Weight, y2, col=6)
segments(1500, max(y2),  2567.5, max(y2), col=1, lwd=2)
segments(2567.5, 24.43478, 3087.5, 24.43478, col=1, lwd=2)
segments(3087.5, min(y2), 4000, min(y2), col=1, lwd=2)
title("Second split")
with(car.test.frame, plot(Mileage~Weight, pch=16))
abline(v=2567.5, col=4, lwd=1.5) 
abline(v=3087.5, col=4, lwd=1.5) 
abline(v=2747.5, col=4, lwd=1.5) 
#points(car.test.frame$Weight, y3, col=6)
segments(1500, max(y3),  2567.5, max(y3), col=1, lwd=2)
segments(2567.5, 25.625, 2747.5, 25.625, col=1, lwd=2)
segments(2747.5, 23.8, 3087.5, 23.8, col=1, lwd=2)
segments(3087.5, min(y3), 4000, min(y3), col=1, lwd=2)
title("Final split")
# the means of the various groups are represented

par(mfrow=c(1,1))
plot(car.tree)
text(car.tree, cex=1.3)

############################
# Example: baseball data
############################
# Two explicative variables
hitter.tree <-  rpart(log(Salary) ~ Hits+Years, data=Hitters, method="anova", subset = complete.cases(Hitters),control=list(cp=0.1))
plot(hitter.tree)
text(hitter.tree,cex=0.5)

plot(Hits ~ Years, data=Hitters, subset = complete.cases(Hitters),pch=20)
segments(4.5,-5,4.4,300,col=4,lwd=2)
segments(4.5,117.5,30,117.5,col=4,lwd=2)
text(1.5,200,labels = 'R1',cex=1)
text(22,50,labels = 'R2',cex=1)
text(22,200,labels = 'R3',cex=1)

# Using more esplicative variables.
set.seed(11)
hitter.tree1 <-  rpart(log(Salary) ~ Hits+Years+RBI+PutOuts+Walks+Runs, data=Hitters, method="anova", subset = complete.cases(Hitters),control=list(cp=0.005))
plot(hitter.tree1)
text(hitter.tree1,cex=0.5)

hitter.tree1$cptable
plotcp(hitter.tree1,lwd=2)
lines(1:12,hitter.tree1$cptable[,3],col=2,type="o",lwd=2)
cverror<-hitter.tree1$cptable[,4]
minxe <- which.min(cverror) # albero con cverror piu' basso
minxe
abline(h=hitter.tree1$cptable[minxe, 4] + hitter.tree1$cptable[minxe, 5],col=4)
ind <- which.max(cverror  <= hitter.tree1$cptable[minxe, 4] + hitter.tree1$cptable[minxe, 5]) # albero piu' piccolo con cverror <= cverror_min + sd cverror_min (1sd rule) 
ind
cpind <- hitter.tree1$cptable[ind,1]
cpind
cpopt <- hitter.tree1$cptable[minxe,1]
cpopt

hitter.tree2 <-  rpart(log(Salary) ~ Hits+Years+RBI+PutOuts+Walks+Runs, data=Hitters, method="anova", subset = complete.cases(Hitters),control=list(cp=cpind))
plot(hitter.tree2)
text(hitter.tree2,cex=1)

##################################
# Example: the spam data set
##################################
set.seed(11)
spam.rpart <- rpart(yesno~crl.tot+dollar+bang+money+n000+make, method="class",
                    data=spam7, cp=0.0005)
printcp(spam.rpart)
plotcp(spam.rpart,lwd=2)
tab <- spam.rpart$cptable
cverror<-tab[,4]
minxe <- which.min(cverror) # albero con cverror piu' basso
minxe
abline(h=tab[minxe, 4] + tab[minxe, 5],col=4)
ind <- which.max(cverror  <= tab[minxe, 4] + tab[minxe, 5]) # albero piu' piccolo con cverror <= cverror_min + sd cverror_min (1sd rule) 
ind

cpind <- tab[ind,1]
cpind
cpopt <- tab [minxe,1]
cpopt
rooter <- min(as.vector(table(spam.rpart$y)/nrow(spam7)))
rooter
rounder <- round(1000 * tab[ind,4] * rooter)/10
rounder

spam.rpartcp <- rpart(yesno~crl.tot+dollar+bang+money+n000+make, method="class",
                    data=spam7, cp=cpind)
plot(spam.rpartcp)
text(spam.rpartcp,cex=0.5)

######################
# Ensemble methods
######################
##########################
# Example: baseball data
##########################
Hitters.compl <- Hitters[complete.cases(Hitters),] # dataframe con casi completi
# bagging
set.seed(11)
hitter.bag <- randomForest(log(Salary) ~ Hits+Years+RBI+PutOuts+Walks+Runs, data=Hitters.compl, mtry=6, importance=TRUE)
hitter.bag
plot(hitter.bag)

# random forest con m=3
set.seed(11)
hitter.rf3 <- randomForest(log(Salary) ~ Hits+Years+RBI+PutOuts+Walks+Runs, data=Hitters.compl, mtry=3, importance=TRUE)
hitter.rf3
plot(hitter.rf3)

# m ottimale rispetto a OOB
set.seed(1)
training.resp<-log(Hitters.compl$Salary)
training.regr<-subset(Hitters.compl, select= c(Hits,Years,RBI,PutOuts,Walks,Runs))
tuneRF(training.regr, training.resp, mtryStart=6)

# random forest con m=4
set.seed(11)
hitter.rf4 <- randomForest(log(Salary) ~ Hits+Years+RBI+PutOuts+Walks+Runs, data=Hitters.compl, mtry=4, importance=TRUE)
hitter.rf4
plot(hitter.rf4)

# valutazione rilevanza regressori
importance(hitter.rf3)
varImpPlot(hitter.rf3)

# boosting d=3
set.seed(11)
hitter.boost3 <- gbm(log(Salary) ~ Hits+Years+RBI+PutOuts+Walks+Runs, data=Hitters.compl, distribution = "gaussian", n.trees = 5000,interaction.depth = 3)
summary(hitter.boost3) # plot con valutazione rilevanza regressori

# boosting d=1 (decision stamp)
set.seed(11)
hitter.boost1 <- gbm(log(Salary) ~ Hits+Years+RBI+PutOuts+Walks+Runs, data=Hitters.compl, distribution = "gaussian", n.trees = 5000,interaction.depth = 1)
summary(hitter.boost1) # plot con valutazione rilevanza regressori

### stima CV del testMSE
# CV testMSE bagging 
predfun.bag <- function(train.x, train.y, test.x, test.y)
  {
      fit.bag <- randomForest(train.x, train.y, mtry=6, importance=TRUE)
      yhat.bag <- predict(fit.bag,newdata=test.x)
      mean((yhat.bag-test.y)^2)
}

set.seed(22)
msecv.bag <- crossval(predfun.bag,as.matrix(training.regr), training.resp, B=1, verbose = F)
msecv.bag$stat
msecv.bag$stat.se

# CV testMSE random forest m=3
predfun.rf3 <- function(train.x, train.y, test.x, test.y){
   
   fit.rf3 <- randomForest(train.x, train.y, mtry=3, importance=TRUE)
   yhat.rf3 <- predict(fit.rf3,newdata=test.x)
   mean((yhat.rf3-test.y)^2)
}
 
set.seed(22)
msecv.rf3 <- crossval(predfun.rf3,as.matrix(training.regr), training.resp, B=1, verbose = F)
msecv.rf3$stat
msecv.rf3$stat.se
 
# CV testMSE random forest m=4
predfun.rf4 <- function(train.x, train.y, test.x, test.y){
 
 fit.rf4 <- randomForest(train.x, train.y, mtry=4, importance=TRUE)
 yhat.rf4 <- predict(fit.rf4,newdata=test.x)
 mean((yhat.rf4-test.y)^2)
}
 
set.seed(22)
msecv.rf4 <- crossval(predfun.rf4,as.matrix(training.regr), training.resp, B=1, verbose = F)
msecv.rf4$stat
msecv.rf4$stat.se
 
# CV testMSE boosting d=3
predfun.boost3 <- function(train.x, train.y, test.x, test.y){
   
 fit.boost <- gbm.fit(train.x, train.y, distribution = "gaussian", n.trees = 5000,interaction.depth = 3, verbose = F)
 yhat.boost <- predict(fit.boost,newdata=test.x,n.trees=5000)
 mean((yhat.boost-test.y)^2)
}
 
set.seed(22)
msecv.boost3 <- crossval(predfun.boost3,as.matrix(training.regr), training.resp, B=1, verbose = F)
msecv.boost3$stat
msecv.boost3$stat.se

# CV testMSE boosting d=1
predfun.boost1 <- function(train.x, train.y, test.x, test.y){
 
 fit.boost <- gbm.fit(train.x, train.y, distribution = "gaussian", n.trees = 5000, interaction.depth = 1, verbose = F)
 yhat.boost <- predict(fit.boost,newdata=test.x,n.trees=5000)
 mean((yhat.boost-test.y)^2)
}

set.seed(22)
msecv.boost1 <- crossval(predfun.boost1,as.matrix(training.regr), training.resp, B=1, verbose = F)
msecv.boost1$stat
msecv.boost1$stat.se

# CV testMSE regression tree ottimale
set.seed(11)
hitter.tree1 <-  rpart(log(Salary) ~ Hits+Years+RBI+PutOuts+Walks+Runs, data=Hitters, method="anova", subset = complete.cases(Hitters),control=list(cp=0.005))
cverror<-hitter.tree1$cptable[,4]
minxe <- which.min(cverror) # albero con cverror piu' basso
hitter.tree1$cptable[minxe, 4]
hitter.tree1$cptable[minxe, 5]

###################################
# Example: the spam data set
###################################
######### trees
?J48
tree <- J48(yesno~crl.tot+dollar+bang+money+n000+make, data = spam7,
            control=Weka_control(M=10))
plot(tree)

# M=10 minimum number of observations in the final groups/nodes
cv.tree <-  evaluate_Weka_classifier(tree, numFolds=10, seed=211)$confusionMatrix # basata si CV_10
############## boosting
?AdaBoostM1
boost <- AdaBoostM1(yesno~crl.tot+dollar+bang+money+n000+make, data = spam7,
                     control = Weka_control(W = list(J48, M=10)))
cv.boost <-  evaluate_Weka_classifier(boost, numFolds=10, seed=1332)$confusionMatrix # basata si CV_10
##############  bagging
set.seed(11)
?randomForest
bag <- randomForest(yesno~crl.tot+dollar+bang+money+n000+make, data = spam7, mtry=6)
cv.bag <- bag$confusion[1:2,1:2] # calcolata usando OOB data (asintoticamente equivalente a CV_n)
##############  RF
set.seed(11)
rf <- randomForest(yesno~crl.tot+dollar+bang+money+n000+make, data = spam7)
cv.rf <- rf$confusion[1:2,1:2] # calcolata usando OOB data (asintoticamente equivalente a CV_n)
ev.acc <- function(CM)
{
  acc <- (CM[1,1] + CM[2,2]) /sum(CM)
  tn <-  CM[1,1] / (CM[1,1]+CM[1,2]) 
  tp <-  CM[2,2] / (CM[2,1]+CM[2,2]) 
  return(c(acc,tn,tp))
}
tabe <- rbind(ev.acc(cv.tree), ev.acc(cv.boost),
             ev.acc(cv.bag), ev.acc(cv.rf)) 
colnames(tabe)<-c("tot. accuracy","true negative","true positive")
rownames(tabe)<- c("Trees","Boosting","Bagging","Random Forests")
tabe
