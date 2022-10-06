######### Lab 5 - Towards multiple linear regression and logistic regression #############


## Install the packages "DAAG", "MASS", "lattice", "compositions", "ISLR", "car" ##

# install.packages("DAAG")
# install.packages("MASS")
# install.packages("lattice")
# install.packages("compositions")
# install.packages("ISLR")
# install.packages("car")



### Multiple linear regression with continuous responses ###

# Example: book weight #

library(DAAG)
library(lattice)
pairs(allbacks[,1:3],lwd=2)
splom(~allbacks[,1:3],lwd=2, groups = cover,auto.key = T, data = allbacks)

allbacks.lm <- lm(weight  ~ volume + area, data=allbacks)
summary(allbacks.lm)

anova(allbacks.lm)

cor(allbacks$volume,allbacks$area)

library(car)
car::vif(allbacks.lm)
DAAG::vif(allbacks.lm)

par(mfrow=c(2,2))
plot(allbacks.lm, which = 1, lwd=2, pch = 16, cex.caption=1)
plot(allbacks.lm, which = 2, xlab="Theoretical quantiles",lwd=2, pch = 16, cex.caption=1)
plot(allbacks.lm, which = 3,lwd=2, pch = 16, cex.caption=1)
plot(allbacks.lm, which = 4,lwd=2, pch = 16, cex.caption=1)
par(mfrow=c(1,1))

# Example: hill races #

library(DAAG)
library(lattice)
gr1 <- splom(~nihills[,c("dist","climb","time")],lwd=2)
gr2 <- splom(~log(nihills[,c("dist","climb","time")]),
             varnames=c("log dist", "log climb", "log time"),lwd=2)
print(gr1, position=c(0, 0, 0.5, 1))
print(gr2, position=c(0.5, 0, 1, 1),  newpage=FALSE)

nihills.lm <- lm(log(time) ~ log(dist) + log(climb), data = nihills)
summary(nihills.lm)

par(mfrow=c(2,2))
plot(nihills.lm, which = 1, lwd=2, pch = 16, cex.caption=1)
plot(nihills.lm, which = 2, xlab="Theoretical quantiles",lwd=2, pch = 16, 
     cex.caption=1)
plot(nihills.lm, which = 3,lwd=2, pch = 16, cex.caption=1)
plot(nihills.lm, which = 4,lwd=2, pch = 16, cex.caption=1)
par(mfrow=c(1,1))

lognihills <- log(nihills)
names(lognihills) <- paste("log", names(nihills), sep="")
str(lognihills)

lognihills$logGrad <- with(nihills, log(climb/dist))
nihillsG.lm <- lm(logtime ~ logdist + logGrad, data=lognihills)
summary(nihillsG.lm)

with(lognihills, cor(cbind(logGrad, logclimb), logdist))

nihillsG.lm_demean <- lm(logtime ~ I(logdist-mean(logdist)) + 
        I(logGrad-mean(logGrad)), data=lognihills)
summary(nihillsG.lm_demean)

nihills.lm <- lm(logtime ~ logdist + logclimb, data=lognihills)
par(mfrow=c(1,2))
termplot(nihills.lm, terms=1,partial.resid=TRUE, lwd.term=2,lwd.se=2,pch=20,
         smooth=panel.smooth, col.smth='blue', col.res="gray30", se=T)
termplot(nihills.lm, terms=2,partial.resid=TRUE, lwd.term=2,lwd.se=2,pch=20,
         smooth=panel.smooth, col.smth='blue', col.res="gray30")
par(mfrow=c(1,1))

par(mfrow=c(1,3))
plot(nihills.lm, which=4, lwd=2, pch = 16, cex.caption=0.8)
plot(nihills.lm, which=5, lwd=2, pch = 16, cex.caption=0.8)
plot(nihills.lm, which=6, lwd=2, pch = 16, cex.caption=0.8)
par(mfrow=c(1,1))

dfbetas(nihills.lm)
influence.measures(nihills.lm)

nihills.lm <- lm(logtime ~ logdist + logclimb, data = lognihills)
nihills2.lm <- lm(logtime ~ logdist + logclimb + I(logdist^2), data = lognihills)
summary(nihills.lm)
summary(nihills2.lm)

AIC(nihills.lm,nihills2.lm)
BIC(nihills.lm,nihills2.lm)

# Example: cars #

plot(dist ~ speed, data = cars, xlim=c(3,1.04*max(speed)),
     ylim=c(0,1.04*max(dist)), xlab = 'Speed', ylab = 'Distance', pch = 16)

cars2.lm <- lm(dist ~ speed + I(speed^2),data=cars)

dat <- data.frame(speed = seq(3,25,length=100))
fv <- predict(cars2.lm,newdata=dat,se=TRUE)
fitvalues <- cars2.lm$coef[1] + cars2.lm$coef[2]*seq(3,25,length=100) + 
       cars2.lm$coef[3]*seq(3,25,length=100)^2
fv$fit
fitvalues

plot(dist ~ speed, data = cars, xlim=c(3,1.04*max(speed)),
     ylim=c(0,1.04*max(dist)), xlab = 'Speed', ylab = 'Distance', pch = 16)
lines(dat$speed,fv$fit,lwd=2, col='red')
with(cars, lines(lowess(dist ~ speed, f=.7), lwd=2, col='blue'))

summary(cars2.lm)

library(DAAG)
vif(cars2.lm)

par(mfrow=c(2,2))
cars2.lm <- lm(dist ~ speed + I(speed^2),data=cars)
plot(cars2.lm, which = 1, lwd=2, pch = 16, cex.caption=0.8)
plot(cars2.lm, which = 2, xlab="Theoretical quantiles",lwd=2, pch = 16, 
     cex.caption=0.8)
plot(cars2.lm, which = 3,lwd=2, pch = 16, cex.caption=0.8)
plot(cars2.lm, which = 4,lwd=2, pch = 16, cex.caption=0.8)
par(mfrow=c(1,1))

cars2w.lm <- lm(dist ~ speed + I(speed^2),data=cars,weights=1/speed)
summary(cars2w.lm)

cars0.lm <- lm(dist ~ speed, data = cars)
cars1w.lm <- lm(dist ~ I(speed^2) -1, data=cars, weights=1/speed)
cars2w.lm <- lm(dist ~ speed + I(speed^2), data=cars, weights=1/speed)
anova(cars0.lm,cars2w.lm)
anova(cars1w.lm,cars2w.lm)

drop1(cars2w.lm, test = "F")

AIC(cars0.lm,cars1w.lm,cars2w.lm)
BIC(cars0.lm,cars1w.lm,cars2w.lm)

summary(cars0.lm)
summary(cars1w.lm)
summary(cars2w.lm)


### Covariates: selection and multicollinearity ###

## Example: coxite ##

library(compositions)
data(Coxite)
coxite <- as.data.frame(Coxite)
pairs(coxite)

coxiteAll.lm <- lm(porosity ~ A+B+C+D+E+depth, data=coxite)
summary(coxiteAll.lm)

coxite0.lm <- lm(porosity ~ A+B+C+D+depth, data=coxite)
summary(coxite0.lm)

library(DAAG)
vif(coxite0.lm)
cor(coxite$porosity, coxite[,-7])

coxite1.lm <- lm(porosity ~ A+B+C, data=coxite)
summary(coxite1.lm)
vif(coxite1.lm)

coxite2.lm <- lm(porosity ~ B+C, data=coxite)
summary(coxite2.lm)
vif(coxite2.lm)
AIC(coxite0.lm,coxite2.lm)


### Factors as explanatory variables ###

## Example: paper resistance ##

paper <- data.frame(resistance =
c(7, 8, 15,  11, 9, 10,# 5%
12, 17, 13, 18, 19, 15,# 10%
14, 18, 19, 17, 16, 18,# 15%
19, 25, 22, 23, 18, 20), # 20%
trt = rep(c("5%", "10%", "15%", "20%"),
c(6, 6, 6, 6)))

paper$trt <- factor(paper$trt,levels=c("5%", "10%", "15%", "20%"))
# paper$trt <- relevel(paper$trt, ref="5%") # an alternative in order to specify the reference level

paper.aov <- aov( resistance ~ trt , data=paper)
summary(paper.aov)

paper.lm1 <- lm(resistance ~ trt,data=paper)
summary(paper.lm1)

summary.lm(paper.aov)


## Example: warp breaks ##

library(lattice)
data(warpbreaks)
stripplot(breaks ~ tension | wool, warpbreaks, cex=1.2, pch=16)

breaks.aov<-aov(sqrt(breaks) ~ tension*wool, warpbreaks)
anova(breaks.aov)

breaks0.aov<-aov(sqrt(breaks) ~ 1, warpbreaks)
breaks1.aov<-aov(sqrt(breaks) ~ tension, warpbreaks)
breaks2.aov<-aov(sqrt(breaks) ~ tension+wool, warpbreaks)
anova(breaks0.aov,breaks1.aov,breaks2.aov,breaks.aov)

summary.lm(breaks.aov)
anova(breaks0.aov,breaks.aov)

## Example: leaf and air temperatures ##

library(DAAG)
library(lattice)
xyplot(tempDiff ~ vapPress, leaftemp, groups=CO2level, pch=19, 
       key=simpleKey(text=c('low','medium','high'),space="top",
       columns=3,points=FALSE,col=c('blue','magenta','darkgreen'),cex=1.2), 
       type=c("p","r"),lwd=2)

leaf.lm1 <- lm(tempDiff ~ 1 , data = leaftemp)
leaf.lm2 <- lm(tempDiff ~ vapPress, data = leaftemp)
leaf.lm3 <- lm(tempDiff ~ CO2level + vapPress, data = leaftemp)
leaf.lm4 <- lm(tempDiff ~ CO2level + vapPress + vapPress:CO2level, data = leaftemp)
anova(leaf.lm1, leaf.lm2, leaf.lm3, leaf.lm4)

summary(leaf.lm3)

par(mfrow=c(2,2))
plot(leaf.lm3, which = 1, lwd=2, pch = 16, cex.caption=0.8)
plot(leaf.lm3, which = 2, lwd=2, pch = 16, cex.caption=0.8)
plot(leaf.lm3, which = 3, lwd=2, pch = 16, cex.caption=0.8)
plot(leaf.lm3, which = 4, lwd=2, pch = 16, cex.caption=0.8)
par(mfrow=c(1,1))

plot1 <- xyplot(tempDiff ~ vapPress, leaftemp, groups=CO2level,pch=19,
abline = list(a=2.6849,b=-0.8392,col='blue',lwd=2),
key=simpleKey(text=c('low','medium','high'),space="top", columns=3,
points=FALSE,col=c('blue','magenta','darkgreen'),cex=1.2), lwd=2)

plot2 <- xyplot(tempDiff ~ vapPress, leaftemp, groups=CO2level,pch=19,
abline = list(a=2.6849+0.3199,b=-0.8392,col='magenta',lwd=2),
key=simpleKey(text=c('low','medium','high'),space="top", columns=3,
points=FALSE,col=c('blue','magenta','darkgreen'),cex=1.2), lwd=2)

plot3 <- xyplot(tempDiff ~ vapPress, leaftemp, groups=CO2level,pch=19,
abline = list(a=2.6849+0.7931,b=-0.8392,col='darkgreen',lwd=2),
key=simpleKey(text=c('low','medium','high'),space="top", columns=3,
points=FALSE,col=c('blue','magenta','darkgreen'),cex=1.2), lwd=2)

plot1+plot2+plot3


### Regression models with discrete responses ###

## Example: teaching program ##

program <- matrix(c(1,2.66,20,0,0,2,2.89,22,0,0,3,3.28,24,0,0,4,2.92,12,0,
                    0,5,4.00,21,0,1,6,2.86,17,0,0,7,2.76,17,0,0,8,2.87,21,
                    0,0,9,3.03,25,0,0,10,3.92,29,0,1,11,2.63,20,0,0,12,3.32,
                    23,0,0,13,3.57,23,0,0,14,3.26,25,0,1,15,3.53,26,0,0,16,
                    2.74,19,0,0,17,2.75,25,0,0,18,2.83,19,0,0,19,3.12,23,1,
                    0,20,3.16,25,1,1,21,2.06,22,1,0,22,3.62,28,1,1,23,2.89,
                    14,1,0,24,3.51,26,1,0,25,3.54,24,1,1,26,2.83,27,1,1,27,
                    3.39,17,1,1,28,2.67,24,1,0,29,3.65,21,1,1,30,4.00,23,1,
                    1,31,3.10,21,1,0,32,2.39,19,1,1), nrow=32, byrow=T)
colnames(program) <- c("OBS","GPA","TUCE","PSI","GRADE")
program <- as.data.frame(program)

mod0.lm <- lm(GRADE ~ GPA + TUCE + PSI , data = program)
plot(program$GRADE,fitted(mod0.lm),pch=19,ylim=c(-0.1,1.1),xlab="GRADE",
     ylab="Fitted values")
abline(0,0,col='red',lwd=2)
abline(1,0,col='red',lwd=2)

boxplot(fitted(mod0.lm)~program$GRADE,xlab="GRADE",ylim=c(-0.1,1.1),
        ylab="Fitted values")
abline(0,0,col='red',lwd=2)
abline(1,0,col='red',lwd=2)

mod.glm <- glm(GRADE ~ PSI , family = binomial, data = program)
summary(mod.glm)

conttable <- table(program$PSI, program$GRADE)
conttable
odds <- conttable[,2]/conttable[,1]
odds
log(odds)
OR <- odds[2]/odds[1]
OR

mod.glm$coef[1]
mod.glm$coef[1]+mod.glm$coef[2]
mod.glm$coef[2]
log(OR)

mod.glm.all <- glm(GRADE ~ PSI + TUCE + GPA, family = binomial, data = program)
summary(mod.glm.all)

plot(program$GRADE,mod.glm.all$fitted.values,pch=19,ylim=c(-0.1,1.1),xlab="GRADE",
     ylab="Fitted values")
abline(0,0,col='red',lwd=2)
abline(1,0,col='red',lwd=2)

# the following commands give the same output
# mod.glm.all$fitted.values
# exp(predict(mod.glm.all))/(1+exp(predict(mod.glm.all)))

pred <- as.numeric(exp(predict(mod.glm.all))/(1+exp(predict(mod.glm.all)))>0.5)
table(pred,program$GRADE)
(18+8)/(18+3+3+8)

library(DAAG)
CVbinary(mod.glm.all)

## Example: credit card ##

library(ISLR)
library(lattice)
attach(Default)
xyplot(income ~ balance, groups=default,data=Default,auto.key=list(columns=2))

par(mfrow=c(1,2))
boxplot(balance~default, data=Default,xlab="default",ylab="balance",
        col=c("deepskyblue","magenta"))     
boxplot(income~default, data=Default,xlab="default",ylab="income",
        col=c("deepskyblue","magenta"))
par(mfrow=c(1,1))

credit1<-glm(default~balance,family="binomial", data=Default)
summary(credit1)

predict(credit1,data.frame(balance=c(1000,2000)),type = c("response"))

Default$studentD<-0
Default$studentD[Default$student=="Yes"]<-1
credit2<-glm(default~studentD,family="binomial", data=Default)
summary(credit2)

# the same result can be obtained with the folowing command, using the variable student
# credit2<-glm(default~student,family="binomial", data=Default)

predict(credit2,data.frame(studentD=c(1,0)),type = c("response"))

credit3<-glm(default~income+balance+studentD,family="binomial", data=Default)
summary(credit3)

## Example: UCB admissions ##

odds.ratio<-function(x){(x[1,1]*x[2,2])/(x[1,2]*x[2,1])}
apply(UCBAdmissions,3,odds.ratio)
margin.table(UCBAdmissions,c(2,1))
odds.ratio(margin.table(UCBAdmissions,c(2,1)))

UCB <- as.data.frame.table(UCBAdmissions["Admitted", , ])
names(UCB)[3] <- "admit"
UCB$reject <- as.data.frame.table(UCBAdmissions["Rejected", , ])$Freq
UCB$Gender <- relevel(UCB$Gender, ref="Male")
UCB$total <- UCB$admit + UCB$reject
UCB$p <- UCB$admit/UCB$total
UCB

UCB.glm1 <- glm(p ~ Dept+Gender, family=binomial, data=UCB, weight=total)
summary(UCB.glm1)

# the same result can be obtained with the command
# UCB.glm1 <- glm(cbind(admit, reject) ~ Dept+Gender, family=binomial, data=UCB)

1/exp(UCB.glm1$coefficients[7])

anova(UCB.glm1, test="Chisq")

UCB.glm2 <- glm(p ~ Gender+Dept, family=binomial, data=UCB, weight=total)
summary(UCB.glm2)
anova(UCB.glm2, test="Chisq")

UCB.glm <- glm(p ~ Dept*Gender, family=binomial, data=UCB, weight=total)
summary(UCB.glm)

anova(UCB.glm, test="Chisq")

