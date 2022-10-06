######### Lab 6 - Predictive and classification methods #############


## Install the packages "MASS", "lattice", "ISLR", "boot", "ElemStatLearn", "class", "crossval", "verification", "RWeka"  ##

# install.packages("MASS")
# install.packages("lattice")
# install.packages("ISLR")
# install.packages("boot")
# install.packages("ElemStatLearn")
# install.packages("class")
# install.packages("crossval")
# install.packages("verification")
# install.packages("RWeka")


### Example: advertising data ###

# Data set given in the book "An Introduction to Statistical Learning: with Applications in R" 
# by G. James, D. Witten, T. Hastie and R. Tibshirani and saved in the .csv file "Advertising.csv"
# in the working directory

Advertising<-read.csv(file="Advertising.csv",header=TRUE)[,-1]
str(Advertising)

# scatterplot matrix

panel.hist <- function(x, ...)
{
    usr <- par("usr")
    on.exit(par(usr))
    par(usr = c(usr[1:2], 0, 1.5) )
    h <- hist(x, plot = FALSE)
    breaks <- h$breaks
    nB <- length(breaks)
    y <- h$counts
    y <- y/max(y)
    rect(breaks[-nB], 0, breaks[-1], y, col = "cyan", ...)
}

pairs(Advertising, panel = panel.smooth, cex = 1.5, pch = 1, bg = "light blue",
      diag.panel = panel.hist, cex.labels = 2, font.labels = 2)

# first multiple linear regression model

mod.adv <- lm(Sales~TV+Radio+Newspaper, Advertising)
summary(mod.adv)
summary(mod.adv)$sigma^2 # estimated error variance
AIC(mod.adv) # AIC criterion

par(mfrow=c(2,2), pty="s", mar=c(3,2,3,2))
plot(mod.adv)
par(mfrow=c(1,1))

# second multiple linear regreession model

mod.adv1 <- lm(Sales~TV+Radio+I(TV^2)+TV:Radio, Advertising)
summary(mod.adv1)
summary(mod.adv1)$sigma^2 # estimated error variance
AIC(mod.adv1) # AIC criterion

par(mfrow=c(2,2), pty="s", mar=c(3,2,3,2))
plot(mod.adv1)
par(mfrow=c(1,1))

# confidence and prediction intervals

intc <- predict(mod.adv1, newdata=data.frame(TV=100,Radio=20), 
                interval="confidence")
intc

intp <- predict(mod.adv1, newdata=data.frame(TV=100,Radio=20), 
                interval="prediction")
intp


### Example: automobile bodily injury claims ###

# Data set given in the book "Regression Modeling with Actuarial and Financial Applications" 
# by E.W. Frees and saved in the .csv file "AutoBI.csv" in the working directory


autobi=read.csv(file="AutoBI.csv",header=TRUE)[,-1]
str(autobi)

autobi$ATTORNEY <- factor(autobi$ATTORNEY)
levels(autobi$ATTORNEY) <- c("yes","no")
autobi$CLMSEX <- factor(autobi$CLMSEX)
levels(autobi$CLMSEX) <- c("male","female")
autobi$MARITAL <- factor(autobi$MARITAL)
levels(autobi$MARITAL) <- c("M","S","W","D")
autobi$CLMINSUR <- factor(autobi$CLMINSUR)
levels(autobi$CLMINSUR) <- c("yes","no")
autobi$SEATBELT <- factor(autobi$SEATBELT)
levels(autobi$SEATBELT) <- c("yes","no")

autobi$AGECLASS <- cut(autobi$CLMAGE, breaks=c(0,18,26,36,47,95))
levels(autobi$AGECLASS) <- c("1","2","3","4","5")
str(autobi)

summary(autobi)

# exploratory data analysis

library(MASS)
par(mfrow=c(1,3), pty="s")
hist.scott(autobi$LOSS, main="LOSS", xlab="", col="lightblue")
rug(autobi$LOSS, col="lightblue")
hist.scott(autobi$LOSS[-which.max(autobi$LOSS)], main="LOSS (without max value)",
           xlab="",col="lightblue")
rug(autobi$LOSS[-which.max(autobi$LOSS)],col="lightblue")
hist.scott(log(autobi$LOSS), main="log(LOSS)",xlab="",col="lightblue")
rug(log(autobi$LOSS),col="lightblue")

par(cex.axis=1.3,cex.lab=1.3, mar=c(5,3,1.5,1))
layout(matrix(c(1:7,7),byrow=TRUE,nrow=2))
ind <- c(1,2,3,4,5,8)
for (i in 1:6)
   barplot(table(autobi[,ind[i]]) , xlab=names(autobi)[ind[i]], 
                col="lightblue")
hist(autobi$CLMAGE,col="lightgreen",xlab="CLMAGE",ylab="",main="",border="white")

library(lattice)
bwplot(log(LOSS)~ ATTORNEY + CLMSEX + MARITAL + CLMINSUR + SEATBELT + AGECLASS,
       data=autobi, ylab="log(LOSS)", outer = TRUE, scales = list(x = "free"),
       xlab="", layout=c(3,2), main="", aspect="fill")    

par(cex.axis=1.3,cex.lab=1.3, mar=c(5,3,1.5,1))
layout(matrix(c(1:7,7),byrow=TRUE,nrow=2))
ind <- c(1,2,3,4,5,8)
for (i in 1:6)
   boxplot(log(autobi$LOSS)~ autobi[,ind[i]] , xlab=names(autobi)[ind[i]], 
                col="lightblue")
plot(autobi$CLMAGE,log(autobi$LOSS),xlab="CLMAGE",ylab="",main="",pch=20)

library(lattice)
densityplot(~log(LOSS), group=ATTORNEY, data=autobi, lwd=2,
       xlab="log(LOSS)",  plot.points=FALSE, auto.key=list(columns=2))

# first multiple linear regression model for log(LOSS)

mod <- lm( log(LOSS) ~ ATTORNEY + CLMAGE, autobi)
summary(mod)

par(mfrow=c(2,2), pty="s", mar=c(3,2,3,2))
plot(mod)
par(mfrow=c(1,1))

# prediction intervals

ci_yes<-predict(mod, newdata=data.frame(ATTORNEY="yes",CLMAGE=30,SEATBELT="yes"), 
                interval="confidence")
ci_yes # age=30 and attorney=yes, log scale
exp(ci_yes) # age=30 and attorney=yes, original scale
ci_no<-predict(mod, newdata=data.frame(ATTORNEY="no",CLMAGE=30,SEATBELT="yes"), 
               interval="confidence")
ci_no # age=30 and attorney=no, log scale
exp(ci_no) # age=30 and attorney=no, original scale

# second multiple linear regression model for log(LOSS)

mod2s <- lm(log(LOSS) ~ATTORNEY + CLMAGE + I(CLMAGE^2) + SEATBELT, 
            data=autobi,  subset = complete.cases(autobi))
summary(mod2s)
AIC(mod2s)
AIC(mod)

# cross-validation and test MSE

library(boot)
autobi_complete<-autobi[complete.cases(autobi),] # data frame with complete observations
mod_bis <- glm(log(LOSS) ~ATTORNEY + CLMAGE, data=autobi_complete)
mod2s_bis <- glm(log(LOSS) ~ATTORNEY + CLMAGE + I(CLMAGE^2) + SEATBELT, 
                 data=autobi_complete)
sum(resid(mod_bis)^2)/length(autobi_complete[,1]) # training MSE for mod_bis
cv.glm(autobi_complete,mod_bis)$delta[2] # cv estimate test MSE for mod_bis
sum(resid(mod2s_bis)^2)/length(autobi_complete[,1]) # training MSE for mod2s_bis
cv.glm(autobi_complete,mod2s_bis)$delta[2] # cv estimate test MSE for mod2s_bis

# confidence and prediction intervals

intc <- predict(mod2s, newdata=data.frame(ATTORNEY="yes",CLMAGE=30,
                SEATBELT="yes"), interval="confidence")
intc # log scale
exp(intc) # original scale
intp <- predict(mod2s, newdata=data.frame(ATTORNEY="yes",CLMAGE=30,
                SEATBELT="yes"), interval="prediction")
intp # log scale
exp(intp) # original scale

intp <- predict(mod2s, newdata=data.frame(ATTORNEY="yes",CLMAGE=30,
                SEATBELT="yes"), interval="prediction", pred.var=1.4122)
intp # log scale
exp(intp) # original scale

age <- seq(18, 80, l=20)
matout <- matrix(0, nrow=20, ncol=3)
for(i in 1:length(age))
matout[i,] <- predict(mod2s, newdata=data.frame(ATTORNEY="yes",CLMAGE=age[i],
                SEATBELT="yes"), interval="prediction")

par(mfrow=c(1,2), pty="s")
plot(age, matout[,1], pch=16, type="l", ylim=range(matout), xlab="CLMAGE", 
     ylab="log(LOSS)")
points(age, matout[,1], pch=16)
lines(age, matout[,2], col=2, lwd=2)
lines(age, matout[,3], col=2, lwd=2)
abline(v=30)
plot(age, exp(matout[,1]), pch=16, type="l", ylim=range(exp(matout)), 
     xlab="CLMAGE", ylab="LOSS")
points(age, exp(matout[,1]), pch=16)
lines(age, exp(matout[,2]), col=2, lwd=2)
lines(age, exp(matout[,3]), col=2, lwd=2)
abline(v=30)


### Example: credit scoring ###

# Data set given in the book "Regression: Models, Methods and Applications" by L. Fahrmeir, Th. Kneib, 
# S. Lang and B. Marx and saved in the .txt file "Scoring.txt" in the working directory

Scoring <- read.table(file="Scoring.txt",header=TRUE)
# acc1=1 (no runnig account) acc1=0 (good or bad running account)
# acc2=1 (good running account) acc2=0 (no or bad running account)
Scoring$account <- 1 - Scoring$acc1 + Scoring$acc2
Scoring$account <- factor(Scoring$account)
Scoring$moral <- factor(Scoring$moral)
Scoring$intuse <- factor(Scoring$intuse)
Scoring$y <- factor(Scoring$y)
str(Scoring)

# exploratory data analysis

par(mfrow=c(2,3), pty="s")
par(cex.axis=1.3,cex.lab=1.3,mar=c(3.5,2,1.5,1))
with(Scoring, barplot(table(y),main="y",col="lightblue"))
with(Scoring, barplot(table(account),main="account",col="lightblue"))
with(Scoring, barplot(table(moral),main="moral",col="lightblue"))
with(Scoring, barplot(table(intuse),main="intuse",col="lightblue"))
with(Scoring, hist.scott(duration,main="duration", xlab="", col="lightblue") )
with(Scoring, hist.scott(amount,main="amount", xlab="", col="lightblue") )
par(mfrow=c(1,1))

layout(matrix(c(1:5,5),byrow=TRUE,nrow=2))
par(cex.axis=1.3,cex.lab=1.3,mar=c(3.5,2,1.5,1))
with(Scoring, boxplot(duration~y, main="duration", col="lightblue"))
with(Scoring, boxplot(amount~y, main="amount", col="lightblue"))
with(Scoring, plot(table(y, account),main=""))
with(Scoring, plot(table(y, moral),main=""))
with(Scoring, plot(table(y, intuse),main=""))
par(mfrow=c(1,1))

# multiple logistic regression model

mod1 <- glm(y~account+duration+amount+moral+intuse, family=binomial, data=Scoring)
prediction1 <- ifelse(predict(mod1, Scoring, type='response') > 0.5, 1, 0)
observed <- Scoring$y

# confusion matrix and prediction diagnostic

confusion <- table(prediction1, observed)
colnames(confusion) <- c("creditworthy","not creditworthy")
rownames(confusion) <- c("creditworthy","not creditworthy")
confusion

library(crossval)
cm <- confusionMatrix(Scoring$y, prediction1, negative="0")
cm

de <- diagnosticErrors(cm)
de

# confusion matrix and prediction diagnostic using cross-validation

predfun.glm = function(train.x, train.y, test.x, test.y, negative)
{
  train.data <- data.frame(train.x, train.y)
  glm.fit <- glm(train.y~., binomial, train.data)
  ynew <- predict(glm.fit, newdata=data.frame(test.x, test.y), type="response")
  ynew <- as.numeric(ynew>0.5)
  out <- confusionMatrix(test.y, ynew, negative=negative)
  return(out)
}
set.seed(1992)
cv.out <- crossval(predfun.glm, Scoring[,4:8], Scoring$y, K=10, B=1,
                  negative="0", verbose=FALSE)

de.cv<-diagnosticErrors(cv.out$stat)
de.cv

tabe <- rbind(de[c(1,3,2)],de.cv[c(1,3,2)])
rownames(tabe)<-c("training data","cross-validated")
colnames(tabe)<-c("tot. accuracy","true negative","true positive")
tabe

# ROC curve

library(verification)
roc.plot(as.numeric(Scoring$y)-1, fitted(mod1), 
    xlab='false positive rate = 1-true negative rate', ylab='true positive rate')
segments(1-de[3],-0.05,1-de[3],de[2],lty=2,col=2,lwd=2)
segments(-0.05,de[2],1-de[3],de[2],lty=2,col=2,lwd=2)

# LDA

z <- lda(y ~ amount + duration, Scoring)
prob.lda <- predict(z)$posterior[,2]

# kNN using the library "RWeka"

library(RWeka)
classifier <- IBk(y ~ moral + intuse+ account + amount + duration, data = Scoring, 
                  control = Weka_control(K = 20, X= TRUE)) ###selects K with CV n
prob.knn <- predict(classifier, type="probability")[,2]

ver_mod1<-verify(as.numeric(Scoring$y)-1, fitted(mod1), bins = FALSE, 
            show = FALSE)
ver_lda<-verify(as.numeric(Scoring$y)-1, prob.lda, bins = FALSE, show = FALSE)
ver_knn<-verify(as.numeric(Scoring$y)-1, prob.knn, bins = FALSE, show = FALSE)
roc.plot(ver_mod1,xlab='false positive rate', ylab='true positive rate',lwd=2,
         show.thres = FALSE)
lines.roc(ver_lda, col = 2, lwd = 2)
lines.roc(ver_knn, col = 4, lwd = 2)

tr.knn <- evaluate_Weka_classifier(classifier, numFolds=0)$confusionMatrix 
cv.knn <- evaluate_Weka_classifier(classifier, numFolds=20, 
              seed=1973)$confusionMatrix 

ev.acc <- function(CM)
{
  acc <- (CM[1,1] + CM[2,2]) /sum(CM)
  tn <-  CM[1,1] / (CM[1,1]+CM[1,2]) 
   tp <-  CM[2,2] / (CM[2,1]+CM[2,2]) 
  return(c(acc,tn,tp))
}

tabe1 <- rbind(ev.acc(tr.knn), ev.acc(cv.knn))
rownames(tabe1)<-c("training data","cross-validated")
colnames(tabe1)<-c("tot. accuracy","true negative","true positive")
tabe1


### Example: two-predictors simulated data ###

# Data set "mixture.example" of the package "ElemStatLearn", associated to the book "The Elements of 
# Statistical Learning, Data Mining, Inference, and Prediction" by T. Hastie, J.H. Friedman and R. Tibshirani

library(ElemStatLearn)
x <- mixture.example$x
g <- mixture.example$y
xnew <- mixture.example$xnew
px1 <- mixture.example$px1
px2 <- mixture.example$px2
prob.bayes <- matrix(mixture.example$prob, length(px1), length(px2))

# Bayes decision boundary

contour(px1, px2, prob.bayes, levels=0.5, labels="", xlab="x1", axes=FALSE,
      ylab="x2", main="Bayes decision boundary", col="blue")
points(x, col=ifelse(g==1,"red", "black"))
box()

# logistic and linear classification boundaries 

mod <- glm(g~x, binomial)
beta <- coef(mod) 
mod.lm <- lm(g~x)
beta.lm <- coef(mod.lm) 

contour(px1, px2, prob.bayes, levels=0.5, labels="", xlab="x1", axes=FALSE, 
        ylab="x2", main="Logistic regression boundary", col="blue", lwd=1)
points(x, col=ifelse(g==1,"red", "black"))
abline(a=-beta[1]/beta[3], b=-beta[2]/beta[3], lwd=2)
abline(a=-beta.lm[1]/beta.lm[3]+0.5/beta.lm[3], 
       b=-beta.lm[2]/beta.lm[3], col="red", lwd=2)
box()

# classification using LDA and QDA

x1<-x[,1]
x2<-x[,2]
lda.fit<-lda(g~x1+x2)
lda.fit
# plot(lda.fit)
lda.pred<-predict(lda.fit,newdata=data.frame(x1=xnew[,1],x2=xnew[,2]))
problda<-matrix(1-lda.pred$posterior,length(px1),length(px2))
# 1-posterior probability, since the class labels are switched

qda.fit<-qda(g~x1+x2)
qda.fit
qda.pred<-predict(qda.fit,newdata=data.frame(x1=xnew[,1],x2=xnew[,2]))
probqda<-matrix(1-qda.pred$posterior,length(px1),length(px2))
# 1-posterior probability, since the class labels are switched

contour(px1, px2, prob.bayes, levels=0.5, labels="", xlab="x1", axes=FALSE,
        ylab="x2", main="Linear and quadratic discriminant analysis", col="blue",lwd=2)
contour(px1, px2, problda, levels=0.5,labels="", xlab="", axes=FALSE,
        ylab="", main="", co="red", lwd=2,add=T)
contour(px1, px2, probqda, levels=0.5,labels="", xlab="", axes=FALSE,
        ylab="", main="", lwd=2,add=T)
points(x, col=ifelse(g==1,"red", "black"),pch=16)
box()


# classification with kNN using the library "class"
# k=1

library(class)
mod1 <- knn(x, xnew, g, k=1, prob=TRUE)
prob <- attr(mod1, "prob")
prob <- ifelse(mod1=="1", prob, 1-prob) # since KNN produces the probabilities of the predicted class (0 or 1)
prob1 <- matrix(prob, length(px1), length(px2))

contour(px1, px2, prob1, levels=0.5, labels="", axes=FALSE, 
        main="1-nearest neighbors", xlab="x1", ylab="x2")
gd <- expand.grid(x=px1, y=px2)
contour(px1, px2, prob.bayes, levels=0.5, labels="", xlab="x1",
    ylab="x2", col="blue", lwd=1,add=T)
points(x, col=ifelse(g==1, "red", "black"))
points(gd, pch=".", cex=1.2, col=ifelse(prob1>0.5, "red", "black"))
box()

# k=10

mod10 <- knn(x, xnew, g, k=10, prob=TRUE)
prob <- attr(mod10, "prob")
prob <- ifelse(mod10=="1", prob, 1-prob)
prob1 <- matrix(prob, length(px1), length(px2))
contour(px1, px2, prob1, levels=0.5, labels="", axes=FALSE, 
        main="10-nearest neighbors", xlab="x1", ylab="x2")
points(x, col=ifelse(g==1, "red", "black"))
gd <- expand.grid(x=px1, y=px2)
points(gd, pch=".", cex=1.2, col=ifelse(prob1>0.5, "red", "black"))
contour(px1, px2, prob.bayes, levels=0.5, labels="", xlab="x1", ylab="x2", 
        col="blue", lwd=1,add=T)
box()

# k=50

mod50 <- knn(x, xnew, g, k=50, prob=TRUE)
prob <- attr(mod50, "prob")
prob <- ifelse(mod50=="1", prob, 1-prob)
prob1 <- matrix(prob, length(px1), length(px2))
contour(px1, px2, prob1, levels=0.5, labels="", axes=FALSE, 
        main="50-nearest neighbors",xlab="x1", ylab="x2")
points(x, col=ifelse(g==1, "red", "black"))
gd <- expand.grid(x=px1, y=px2)
points(gd, pch=".", cex=1.2, col=ifelse(prob1>0.5, "red", "black"))
contour(px1, px2, prob.bayes, levels=0.5, labels="", xlab="x1", ylab="x2", 
        col="blue", lwd=1,add=T)
abline(a=-beta[1]/beta[3], b=-beta[2]/beta[3], col=2, lwd=2)
box()

