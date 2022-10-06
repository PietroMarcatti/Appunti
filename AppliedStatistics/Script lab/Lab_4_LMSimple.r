######### LAB 4 - Linear regression with a single predictor #############


## Install the packages "DAAG", "MASS", "lattice"  ##

# install.packages("DAAG")
# install.packages("MASS")
# install.packages("lattice")


### Example: roller data ###

# model fitting #

library(DAAG)
roller.lm <- lm(depression ~ weight,data=roller)
attributes(roller.lm)

plot(depression ~ weight, data = roller, 
     xlim=c(0,1.04*max(weight)),ylim=c(0,1.04*max(depression)),
     xlab = 'Weight of roller', ylab = 'Depression', pch = 16)
roller.lm <- lm(depression ~ weight,data=roller)
abline(roller.lm,col='red',lwd=2)
points(roller$weight,fitted.values(roller.lm), pch = 16, 
       col="red", lwd=2)
segments(roller$weight,roller$depression,roller$weight,
         fitted.values(roller.lm), col="blue", lwd=2)

attributes(summary(roller.lm))
summary(roller.lm)

SEb <- summary(roller.lm)$coefficients[2, 2]
SEb

coef(roller.lm)[2] + qt(c(0.025,.975), 8)*SEb

# confidence intervals and prediction intervals #

roller.lm$fitted.values
roller.pred <- predict(roller.lm, se.fit=TRUE)
roller.pred$fit
roller.pred$se.fit

se.pred <- sqrt(roller.pred$se.fit^2+roller.pred$residual.scale^2)
se.pred

# Step 1
plot(depression ~ weight, data = roller, 
     xlim=c(0,1.04*max(weight)),ylim=c(0,1.04*max(depression)),
xlab = 'Weight of roller', ylab = 'Depression', pch = 16)
roller.lm <- lm(depression ~ weight,data=roller)
abline(roller.lm,col='red',lwd=2)
# Step 2
xy <- data.frame(weight = pretty(seq(1,13,1), 20))
# Step 3
yhat <- predict(roller.lm, newdata = xy, interval="confidence")
ci <- data.frame(lower=yhat[, "lwr"], upper=yhat[, "upr"])
lines(xy$weight, ci$lower, lty = 2, lwd=2, col="red")
lines(xy$weight, ci$upper, lty = 2, lwd=2, col="red")
# Step 4
yhatob <- predict(roller.lm, newdata = xy, interval="prediction")
ciob <- data.frame(lower=yhatob[, "lwr"], upper=yhatob[, "upr"])
lines(xy$weight, ciob$lower, lty = 2, lwd=2)
lines(xy$weight, ciob$upper, lty = 2, lwd=2)

# diagnostic plots and aov table #

par(mfrow=c(1,2))
plot(roller.lm, which = 1, lwd=2, pch = 16, cex.caption=0.8)
plot(roller.lm, which = 2, xlab="Theoretical quantiles",
     lwd=2, pch = 16, cex.caption=0.8)
par(mfrow=c(1,1))

par(mfrow=c(1,2))
plot(roller.lm, which = 3, lwd=2, pch = 16, cex.caption=0.8)
plot(roller.lm, which = 4, lwd=2, pch = 16, cex.caption=0.8)
par(mfrow=c(1,1))

anova(roller.lm)


### Example: paper resistance ###

# ANOVA model #

library(lattice)
paper <- data.frame(resistance =
c(7, 8, 15,  11, 9, 10,# 5%
12, 17, 13, 18, 19, 15,# 10%
14, 18, 19, 17, 16, 18,# 15%
19, 25, 22, 23, 18, 20), # 20%
trt = rep(c("5%", "10%", "15%", "20%"),
c(6, 6, 6, 6)))
paper$trt <- relevel(paper$trt, ref="5%")
paper

stripplot(resistance~trt, aspect=0.6, data=paper, xlab="concentration", 
          ylab="resistance")

paper.aov <- aov(resistance~trt,data=paper) 
anova(paper.aov) # the same result is given by summary(paper.aov)

concentration <- c(5, 5, 5, 5, 5, 5, 10, 10, 10, 10, 10, 10, 15, 15, 15, 15, 15, 
                   15, 20, 20, 20, 20, 20, 20) # treatment specified as numeric vector
plot(paper$resistance ~ concentration,xlab = 'Concentration', ylab = 'Resistance', 
     xlim=c(3,22), ylim=c(4,27), pch = 16)

trt1 <- summary.lm(paper.aov)$coefficients[1,1] # mean first group
trt1
trt2 <- trt1 + summary.lm(paper.aov)$coefficients[2,1] # mean second group
trt2
trt3 <- trt1 + summary.lm(paper.aov)$coefficients[3,1] # mean third group
trt3 
trt4 <- trt1 + summary.lm(paper.aov)$coefficients[4,1] # mean fourth group
trt4

points(c(5,10,15,20), c(trt1,trt2,trt3,trt4), col='red',lwd=2)
lines(c(5,10,15,20), c(trt1,trt2,trt3,trt4), col='red',lwd=2)

# # linear model with a factor predictor #
# 
# paper.lm1 <- lm(resistance ~ trt,data=paper)
# summary(paper.lm1)
# 
# concentration <- c(5, 5, 5, 5, 5, 5, 10, 10, 10, 10, 10, 10, 15, 15, 15, 15, 15, 
#          15, 20, 20, 20, 20, 20, 20) # treatment specified as numeric vector
# plot(concentration,paper$resistance,xlab = 'Concentration', ylab = 'Resistance', 
#     xlim=c(3,22), ylim=c(4,27), pch = 16)
# points(c(5,10,15,20), 
#       c(paper.lm1$coef[1],paper.lm1$coef[1]+paper.lm1$coef[2],
#         paper.lm1$coef[1]+paper.lm1$coef[3],paper.lm1$coef[1]+
#           paper.lm1$coef[4]), col='red',lwd=2)
# lines(c(5,10,15,20), 
#       c(paper.lm1$coef[1],paper.lm1$coef[1]+paper.lm1$coef[2],
#         paper.lm1$coef[1]+paper.lm1$coef[3],paper.lm1$coef[1]+
#           paper.lm1$coef[4]), col='red',lwd=2)

# linear model #

paper.lm2 <- lm(paper$resistance ~ concentration)
summary(paper.lm2)

plot(paper$resistance ~ concentration,xlab = 'Concentration', ylab = 'Resistance', 
     xlim=c(3,22), ylim=c(4,27), pch = 16)
abline(paper.lm2, col='blue',lwd=2)
points(c(5,10,15,20), c(trt1,trt2,trt3,trt4), col='red',lwd=2)
lines(c(5,10,15,20), c(trt1,trt2,trt3,trt4), col='red',lwd=2)


### Example: cars ###

# model fitting #

cars.lm <- lm(dist ~ speed, data = cars)
summary(cars.lm)

plot(dist ~ speed, data = cars, xlim=c(0,1.04*max(speed))
     ,ylim=c(0,1.04*max(dist)),xlab = 'Speed', ylab = 'Distance', pch = 16)
abline(cars.lm,col='red',lwd=2)

# diagnostic plots #

plot(cars.lm, which = 1, lwd=2, pch = 16, cex.caption=0.6)
plot(cars.lm, which = 3, lwd=2, pch = 16, cex.caption=0.6)
plot(cars.lm, which = 2, xlab="Theoretical quantiles",
     lwd=2, pch = 16, cex.caption=0.6)
plot(cars.lm, which = 4,lwd=2, pch = 16, cex.caption=0.6)

plot(dist ~ speed, data = cars, xlim=c(0,1.04*max(speed)),
     ylim=c(0,1.04*max(dist)),
xlab = 'Speed', ylab = 'Distance', pch = 16)
abline(cars.lm,col='red',lwd=2)
with(cars, lines(lowess(dist ~ speed, f=.7), lwd=2, col='blue'))

# model with sqrt(dist) #

sqrtcars.lm <- lm(sqrt(dist) ~ speed, data = cars)
summary(cars.lm)
summary(sqrtcars.lm)

anova(cars.lm)
anova(sqrtcars.lm)

# diagnostic plots #

par(mfrow=c(1,2))
plot(sqrtcars.lm, which=1,lwd=2, pch = 16, cex.caption=0.6)
plot(sqrtcars.lm, which=2, xlab="Theoretical quantiles",
     lwd=2, pch = 16, cex.caption=0.6)

plot(sqrtcars.lm, which=3,lwd=2, pch = 16, cex.caption=0.6)
plot(sqrtcars.lm, which=4,lwd=2, pch = 16, cex.caption=0.6)
par(mfrow=c(1,1))

plot(sqrt(dist) ~ speed, data = cars, xlim=c(3,26),
xlab = 'Speed', ylab = 'sqrt(Distance)', pch = 16)
abline(sqrtcars.lm,col='red',lwd=2)
with(cars, lines(lowess(sqrt(dist) ~ speed, f=.7), 
                 lwd=2, col='blue'))

# Box-Cox transformation #

library(MASS)
lambdares <- boxcox(cars.lm, lambda = seq(0, 1, 0.05))

lambdares <- boxcox(cars.lm, lambda = seq(0, 1, 0.05), plotit=F, interp=F)
lambdares

# confidence and prediction intervals #

# Step 1
plot(sqrt(dist) ~ speed, data = cars, xlim=c(2,27),
     ylim=c(0,1.04*max(sqrt(dist))), xlab = 'Speed', 
     ylab = 'sqrt(Distance)', pch = 16, main="Transfomed data set")
sqrtcars.lm <- lm(sqrt(dist) ~ speed, data=cars)
abline(sqrtcars.lm,col='red',lwd=2) # regression line
xy <- data.frame(speed = pretty(seq(2,28,1), 25))
yhat <- predict(sqrtcars.lm, newdata = xy, interval="confidence") # ci
ci <- data.frame(lower=yhat[, "lwr"], upper=yhat[, "upr"])
lines(xy$speed, ci$lower, lty = 2, lwd=2, col="red")
lines(xy$speed, ci$upper, lty = 2, lwd=2, col="red")
yhatob <- predict(sqrtcars.lm, newdata = xy, interval="prediction") # pi
ciob <- data.frame(lower=yhatob[, "lwr"], upper=yhatob[, "upr"])
lines(xy$speed, ciob$lower, lty = 2, lwd=2)
lines(xy$speed, ciob$upper, lty = 2, lwd=2)
# Step 2
xy <- data.frame(speed = pretty(seq(2,27,1), 25))
yhat <- predict(sqrtcars.lm, newdata = xy, interval="confidence")
yhat <- yhat^2 # converted ci
ci <- data.frame(lower=yhat[, "lwr"], upper=yhat[, "upr"])
yhatob <- predict(sqrtcars.lm, newdata = xy, interval="prediction")
yhatob <- yhatob^2 # converted pi
ciob <- data.frame(lower=yhatob[, "lwr"], upper=yhatob[, "upr"])

# Step 3
plot(dist ~ speed, data = cars, xlim=c(2,27),ylim=c(0,1.04*max(dist)), 
      xlab = 'Speed', ylab = 'Distance', pch = 16, 
      main="Original data set") # original scatterplot
lines(seq(2,27,0.05),(sqrtcars.lm$coefficients[1]+
      seq(2,27,0.05)*sqrtcars.lm$coefficients[2])^2,
      col='red',lwd=2) # converted regression line
lines(xy$speed, ci$lower, lty = 2, lwd=2, col="red") # converted ci
lines(xy$speed, ci$upper, lty = 2, lwd=2, col="red")
lines(xy$speed, ciob$lower, lty = 2, lwd=2) # converted pi
lines(xy$speed, ciob$upper, lty = 2, lwd=2)


### Example: books ###

# model fitting #

library(DAAG)
softbacks.lm <- lm(weight ~ volume, data=softbacks)
plot(softbacks$volume[-c(4,6)], softbacks$weight[-c(4,6)], xlab = 'Volume', 
     ylab = 'Weight', xlim=c(370,1520), ylim=c(230,1100), pch = 16)
points(softbacks$volume[4], softbacks$weight[4], pch=16, lwd=2, col='red')
points(softbacks$volume[6], softbacks$weight[6], pch=16, lwd=2, col='blue')
abline(softbacks.lm,col='red',lwd=2)

summary(softbacks.lm)

# diagnostic plots #

par(mfrow=c(1,2))
plot(softbacks.lm, which = 1, lwd=2, pch = 16, cex.caption=0.7)
plot(softbacks.lm, which = 4, lwd=2, pch = 16, cex.caption=0.7)
par(mfrow=c(1,1))

softbacks.lm <- lm(weight ~ volume, data=softbacks)
plot(softbacks$volume[-c(4,6)], softbacks$weight[-c(4,6)], 
     xlab = 'Volume', ylab = 'Weight', xlim=c(370,1520), 
     ylim=c(230,1100), pch = 16)
points(softbacks$volume[4], softbacks$weight[4], pch=16, 
       lwd=2, col='red')
points(softbacks$volume[6], softbacks$weight[6], pch=16, 
       lwd=2, col='blue')
abline(softbacks.lm,col='black',lwd=2)

for(i in 1:8)
{
  cols <- 'black'
  cols <- ifelse(i==4, 'red', cols)
  cols <- ifelse(i==6, 'blue', cols)
  mod <- lm(weight ~ volume, data=softbacks[-i,])
  abline(mod, lty=2, lwd=2, col=cols)
}
