######### LAB 3b - A review of inference concepts - Statistical inference #############


## Install the package "DAAG"  ##

# install.packages("DAAG")


### Introduction to statistical inference ###

## Example: temperatures ##

hist(nhtemp,freq=F,main=' ',xlab=' ',ylab=' ')
lines(density(nhtemp),lwd=2)
lines(seq(45,60,0.01),dnorm(seq(45,60,0.01),mean(nhtemp),
                            sqrt(var(nhtemp))),col='red',lwd=2)

mean(nhtemp)
median(nhtemp)
var(nhtemp)
sd(nhtemp)

mean((nhtemp-mean(nhtemp))^3)/sqrt(var(nhtemp))^3
mean((nhtemp-mean(nhtemp))^4)/sqrt(var(nhtemp))^4

## Example: roller data ##

library(DAAG)
plot(depression ~ weight, data = roller, 
     xlim=c(0,1.04*max(weight)),ylim=c(0,1.04*max(depression)),
     xlab = 'Weight of roller', ylab = 'Depression', pch = 16)
roller.lm <- lm(depression ~ weight,data=roller)
abline(roller.lm,col='red',lwd=2)

roller.lm$coef

X <- cbind(1, roller$weight)
y <- roller$depression

solve(t(X)%*%X)%*%(t(X)%*%y)


### Basic concepts of point estimation ###

## Example: elastic bands ##

ambient <- c(254, 252, 239, 240, 250, 256, 267, 249, 259, 269)

heated <- c(233, 252, 237, 246, 255, 244, 248, 242, 217, 257, 254)

mean(ambient)-mean(heated)

s2p <- ((var(heated))*(length(heated)-1)
        +(var(ambient))*(length(ambient)-1))/
        (length(heated)+length(ambient)-2)
s2p

sqrt(s2p)*sqrt((1/length(heated))+(1/length(ambient)))

(mean(ambient)-mean(heated))/(sqrt(s2p)*
      sqrt((1/length(heated))+(1/length(ambient))))

## Point estimators and their properties ##

N <- 10000
set.seed(10)
samp <- rexp(N,1/5)
mean(samp)
var(samp)
sd(samp)

set.seed(10)
repl<-10000
n <- 10
sampmean <- NULL
for (i in 1:repl)
{
  sam <- rexp(n,1/5)
  sampmean <- c(sampmean,mean(sam))
  
}

hist(sampmean)
mean(sampmean)
sd(sampmean)

set.seed(10)
repl<-10000
n <- 10
sampvar <- NULL
variance <- NULL
for (i in 1:repl)
{
  sam <- rexp(n,1/5)
  sampvar <- c(sampvar,var(sam))
  variance <- c(variance,var(sam)*9/10)
}

mean(sampvar)
mean(variance)


### Basic concepts of interval estimation ###

## Confidence interval for the mean ##

# Step 1)
set.seed(12)
# Step 2)
flag <- 0 
# Step 3)
y <- rnorm(15)
# Step 4)
ci <-c(mean(y)-qt(0.975,df=14)*sd(y)/sqrt(15),
       mean(y)+qt(0.975,df=14)*sd(y)/sqrt(15))
# Step 5)
if(ci[1]*ci[2]<0){
plot(c(1,1),ci,ylim=c(-1.1,1.2), xlim=c(0,100),
     type='l',xlab=' ',ylab=' ',lwd=2)
flag <- flag+1} 
if(ci[1]*ci[2]>0){
plot(c(1,1),ci,ylim=c(-1.2,1.2), xlim=c(0,100),
     type='l',xlab=' ',ylab=' ',col='red',lwd=2)}
for (i in 2:100){
# Step 3)
y <- rnorm(15)
# Step 4)
ci <-c(mean(y)-qt(0.975,df=14)*sd(y)/sqrt(15),
       mean(y)+qt(0.975,df=14)*sd(y)/sqrt(15))
# Step 5)
if(ci[1]*ci[2]<0){
lines(c(i,i),ci,type='l',lwd=2)
flag <- flag+1
}
if(ci[1]*ci[2]>0){
lines(c(i,i),ci,type='l',col='red',lwd=2)
}
}
# Step 6)
abline(0,0)
# Step 7)
flag/100

## Confidence intervals with hypothesis testing commands ##

set.seed(12)
y <- rnorm(15)
ci <-c(mean(y)-qt(0.975,df=14)*sd(y)/sqrt(15),
       mean(y)+qt(0.975,df=14)*sd(y)/sqrt(15))
print(ci)

t.test(y)

attributes(t.test(y))

t.test(y)$conf.int


### Basic concepts of hypothesis testing ###

# Example: maximum temperature ##

t81 <- c(39,39,40,33,36,40,37,41,39,34,42,41,
         42,44,42,42,39,42,41,40,43,43,40,39,37)
mean(t81)
median(t81)
sd(t81)
sem <- sd(t81)/sqrt(length(t81))
sem
qt(0.025,24,lower.tail = FALSE)

c(mean(t81)-qt(0.025,24,lower.tail = FALSE)*sem,
  mean(t81)+qt(0.025,24,lower.tail = FALSE)*sem)

(mean(t81)-37.5)/sem

t.test(t81,mu=37.5)

par(mfrow=c(1,2))
xx<-seq(-5,5,0.01)
plot(xx,dt(xx,24),type='l',lwd=2,cex.axis=1.3,
     ylim=c(0,0.45),xlab=" ",ylab=" ")
cord.x <- c(-5,seq(-5,-2.064,0.01),-2.064) 
cord.y <- c(0,dt(seq(-5,-2.064,0.01),24),0) 
polygon(cord.x,cord.y,col='skyblue')
cord.x <- c(2.064,seq(2.064,5,0.01),2.064) 
cord.y <- c(0,dt(seq(2.064,5,0.01),24),0) 
polygon(cord.x,cord.y,col='skyblue')
abline(0,0,lwd=2)
lines(4.119,0,type='p',lwd=3,col='red')

xx1<-seq(-5,-4.119,0.01)
xx2<-seq(4.119,5,0.01)
plot(xx1,dt(xx1,24),type='l',lwd=3,cex.axis=1.3,
     xlim=c(-5,5),ylim=c(0,0.45),xlab=" ",ylab=" ",col='red')
lines(xx2,dt(xx2,24),lwd=3,col='red')
yy <- seq(-4.119,4.119,0.01)
lines(yy,dt(yy,24),type='l',lwd=2)
lines(c(-4.119,4.119),c(0,0),lwd=2)
par(mfrow=c(1,1))

## Example: physical activity ##

p <- 108/200
p
se <- sqrt(p*(1-p)/200)
se
z <- (p-0.492)/sqrt(0.492*(1-0.492)/200)
z
z^2
2*pnorm(-abs(z))

prop.test(108, 200, p = 0.492,correct = FALSE)

attributes(prop.test(108, 200, p = 0.492,
                     correct = FALSE))

prop.test(108, 200, p = 0.492,correct = FALSE)$stat

## Example: white and red wines ##

xw <- c(28.4,32.2,37.0,32.4,33.2,18.7,33.7,50.0,49.8,34.5,45.8,33.1,
        24.1,31.0,24.8,19.0,17.5,19.4,24.7,9.9,29.1,18.4,34.7,29.3,
        15.6,20.7,22.2,18.7,11.8,12.1)
xr <- c(7.3,27.9,20.4,18.5,6.6,9.1,1.5,13.9,11.1,34.7,57.0,1.3,17.6,
        6.1,22.9,27.3,30.0,19.6,21.8,18.2,8.6,12.8,18.6,29.4,28.5,
        16.6,30.1,27.2,19.6,16.3,29.9,26.3,26.5,24.3,19.1,28.3,36.8)
mean(xw)
median(xw)
mean(xr)
median(xr)
var(xw)
var(xr)
sd(xw)
sd(xr)

var(xw)
var(xr)
F <- var(xw)/var(xr)
F
2*min(pf(F,length(xw)-1,length(xr)-1),
      pf(F,length(xw)-1,length(xr)-1,lower.tail = FALSE)) # p-value

var.test(xw, xr, ratio = 1)

sem2x <-var(xw)/length(xw)
sem2x
sem2y <-var(xr)/length(xr)
sem2y

s2p <- (var(xw)*(length(xw)-1)+var(xr)*(length(xr)-1))/
  (length(xw)+length(xr)-2)
s2p

sedt <- sqrt(s2p)*sqrt(1/length(xw)+1/length(xr))
sedt
sedw <- sqrt(sem2x+sem2y)
sedw

tt <- (mean(xw)-mean(xr))/sedt # equal variances
tt
tw <- (mean(xw)-mean(xr))/sedw # unequal variances
tw

t.test(xw,xr,var.equal = TRUE)  # equal variances
t.test(xw,xr,var.equal = FALSE)  # unequal variances

## Example: temperatures ##

t80 <- c(36,35,36,34,37,40,37,41,38,32,36,39,36,40,37,37,38,40,37,39,
         39,41,38,38,35)
t81 <- c(39,39,40,33,36,40,37,41,39,34,42,41,42,44,42,42,39,42,41,40,
         43,43,40,39,37)
summary(t80)
summary(t81)

diff <- t80-t81
summary(diff)
sd(diff)/sqrt(length(diff)) # the standard error of the difference

mean(diff)/(sd(diff)/sqrt(length(diff)))

t.test(t80,t81,pair=TRUE)

## Non-parametric testing procedures ##

wilcox.test(xw,xr)

wilcox.test(t80,t81, paired=T)

## Example: labor training program ##

px <- 217/297
px
py <- 65/128
py
p <- (217+65)/(297+128)
p

sed <- sqrt(p*(1-p)*(1/297+1/128))
sed

z <- (px-py)/sed
z

2*pnorm(abs(z),lower.tail = FALSE)

z^2
prop.test(c(217,65),c(297,128), correct = FALSE)

X <- t(matrix(c(217,(297-217), 65,(128-65)),2,2))
colnames(X) <- c("Drop_yes","Drop_no")
row.names(X) <- c("Prog_yes","Prog_no")
X
chisq.test(X, correct=F)

## Testing for correlation ##

set.seed(442)
n <- 100
r <- 0.8
x <- rnorm(100)
y <- r*x + sqrt(1-r^2)*rnorm(n)

ii <- order(-x)
x[ii[1:5]] <- x[ii[1:5]]*3

plot(x,y,xlab=' ',ylab=' ',xlim=c(-3,7),pch=16)
abline(lm(y~x),col=2,lwd=2)
abline(lm(y~x,subset=is.element(1:100,ii[1:5])),col=2,lty=2,lwd=2)
abline(lm(y~x,subset=!is.element(1:100,ii[1:5])),col=2,lty=3,lwd=3)
legend(4,-1, legend=c("Full", "Transformed", "Original"),lty=1:3, col="red")

cor(x,y,method='pearson')
cor(x,y,method='spearman')
cor(x[-ii[1:5]],y[-ii[1:5]],method='pearson')
cor(x[-ii[1:5]],y[-ii[1:5]],method='spearman')

cor.test(x,y,method='pearson')
cor.test(x[-ii[1:5]],y[-ii[1:5]],method='pearson')

cor.test(x,y,method='spearman')
cor.test(x[-ii[1:5]],y[-ii[1:5]],method='spearman')


### Basic concepts of model selection ##

## Example: black cherry trees ##

par(mfrow=c(1,2))
mod1 <- lm(Volume ~ Girth, data = trees)
mod11 <- lm(Volume ~ Height, data = trees)
plot(Volume ~ Girth, data = trees,lwd=2,cex.lab=1.5)
abline(mod1,lwd=2,col='red')
plot(Volume ~ Height, data = trees,lwd=2,cex.lab=1.5)
abline(mod11,lwd=2,col='red')
par(mfrow=c(1,1))

mod2 <- lm(Volume ~ Girth + Height, data = trees)

logLik(mod1)
AIC(mod1)
BIC(mod1)
logLik(mod2)
AIC(mod2)
BIC(mod2)

AIC(mod2, k=log(length(trees$Volume)))

# Initialise the CV index
cv1 <- 0
cv2 <- 0
n <- length(trees$Volume)
i <-1
for (i in 1:n){
# step 1
mod1i <- lm(Volume ~ Girth, data = trees[-i,])
mod2i <- lm(Volume ~ Girth + Height, data = trees[-i,])
# step 2
mu1 <- mod1i$coefficients[1] + mod1i$coefficients[2]*trees$Girth[i]
mu2 <- mod2i$coefficients[1] + mod2i$coefficients[2]*trees$Girth[i] + 
            mod2i$coefficients[3]*trees$Height[i]
sd1 <- sqrt(sum(mod1i$residuals^2)/(n-3))
sd2 <- sqrt(sum(mod2i$residuals^2)/(n-4))
# step 3
cv1 <- cv1 - log(dnorm(trees$Volume[i],mu1,sd1))
cv2 <- cv2 - log(dnorm(trees$Volume[i],mu2,sd2))
}
cv1
cv2


### Contingency tables ###

## Example: steel rods ##

rods <- matrix(c(10, 102, 8, 34, 161, 5, 
                 12, 79, 9, 10, 60, 10),nrow=4,byrow=TRUE)
rods

xtot <- apply(rods,1,sum)
ytot <- apply(rods,2,sum)

xtot <- as.matrix(xtot)
ytot <- as.matrix(ytot)

rods_ind <- xtot%*%t(ytot)/sum(xtot)
rods_ind

chisq_obsstat <- sum((rods-rods_ind)^2/rods_ind)
chisq_obsstat

1-pchisq(chisq_obsstat, 6)

chisq.test(rods)
residuals(chisq.test(rods))

## Example: labor training program ##

dropout <- matrix(c(63, 65, 80, 217), nrow=2, byrow=TRUE)
colnames(dropout) <- c("yes","no")
rownames(dropout) <- c("yes","no")
names(dimnames(dropout)) <- c("program","high school graduate")
dropout

xtot <- apply(dropout,1,sum)
ytot <- apply(dropout,2,sum)
xtot <- as.matrix(xtot)
ytot <- as.matrix(ytot)

xtot%*%t(ytot)/sum(xtot)

chisq.test(dropout,correct =FALSE)

