######### LAB 3a - A review of inference concepts - Statistical models #############


## Install the packages "DAAG", fBasics", "nortest"  ##

# install.packages("DAAG")
# install.packages("fBasics")
# install.packages("nortest")


### Some introductory examples ###

## Example: temperatures

hist(nhtemp,freq=F,main=' ',xlab=' ',ylab=' ')
lines(density(nhtemp),lwd=2)
lines(seq(45,60,0.01),dnorm(seq(45,60,0.01),
        mean(nhtemp),sqrt(var(nhtemp))),col='red',lwd=2)

summary(nhtemp)
mean(nhtemp)
median(nhtemp)
var(nhtemp)

mean((nhtemp-mean(nhtemp))^3)/sqrt(var(nhtemp))^3
mean((nhtemp-mean(nhtemp))^4)/sqrt(var(nhtemp))^4

## Example: roller data

library(DAAG)
plot(depression ~ weight, data = roller, 
     xlim=c(0,1.04*max(weight)),ylim=c(0,1.04*max(depression)),
xlab = 'Weight of roller', ylab = 'Depression', pch = 16)
roller.lm <- lm(depression ~ weight,data=roller)
abline(roller.lm,col='red',lwd=2)

attributes(roller.lm)
roller.lm$coef
attributes(summary(roller.lm))
summary(roller.lm)$coef

OLSfunct <- function(coef, y, x)
{
  return(sum((y-coef[1]-coef[2]*x)^2))
}

param <- c(0,0)
res <- optim(param, OLSfunct, y=roller$depression, x=roller$weight)
res$par

### Basic statistical models ###

## Discrete uniform distribution

xx<-c(1:6)
plot(xx,rep(1/6,6),pch=19,xlim=c(0,7),ylim=c(0,1.1/6),
     cex.axis=1.5,xlab="x",ylab="f(x)")
segments(0,0,7,0,lwd=2)


xx <- letters[1:10]
set.seed(11)
sampvec <- sample(xx, 100, replace=T)
table(sampvec)/sum(table(sampvec))
plot(table(sampvec)/sum(table(sampvec)))
abline(h=0.1, col=2)

set.seed(11)
sampvec <- sample(xx, 10000, replace=T)
table(sampvec)/sum(table(sampvec))
plot(table(sampvec)/sum(table(sampvec)))
abline(h=0.1, col=2)

## Bernoulli distribution

plot(c(0,1),c(2/3,1/3),pch=19,lwd=2,xlab='x',
     ylab='f(x)',xlim=c(-0.1,1.2),ylim=c(0,1))
segments(-0.1,0,1.2,0,lwd=3)

xx <- c(0,1)
set.seed(101)
sampvec <- sample(xx, 1000, prob=c(2/3,1/3), replace=T) 
table(sampvec)/sum(table(sampvec))
plot(table(sampvec)/sum(table(sampvec)), ylim=c(0,1))

set.seed(101)
plot(table(rbinom(1000,1,1/3))/1000, ylim=c(0,1))

## Binomial distribution

par(mfrow=c(2,2)) # Step 1
xx<-seq(0,10,1) # Step 2
plot(xx,dbinom(xx,10,0.2),pch=19,ylim=c(0,0.5),
     cex.axis=1.5,xlab=" ",ylab=" ",main="A) n=10, p=0.2") # Step 3
segments(0,0,10,0,lwd=2) # Step 4

plot(xx,dbinom(xx,10,0.5),pch=19,ylim=c(0,0.5),lwd=2,
     cex.axis=1.5,xlab=" ",ylab=" ",main="B) n=10, p=0.5") # Step 3
segments(0,0,10,0,lwd=2) # Step 4

plot(xx,dbinom(xx,10,0.8),pch=19,ylim=c(0,0.5),lwd=2,
     cex.axis=1.5,xlab=" ",ylab=" ",main="C) n=10, p=0.8") # Step 3
segments(0,0,10,0,lwd=2) # Step 4

xx<-seq(0,20,1) # Step 2
plot(xx,dbinom(xx,20,0.5),pch=19,ylim=c(0,0.5),lwd=2,
     cex.axis=1.5,xlab=" ",ylab=" ",main="D) n=20, p=0.5") # Step 3
segments(0,0,20,0,lwd=2) # Step 4
par(mfrow=c(1,1))

qbinom(seq(0,1,0.1), 10, 0.2)
pbinom(0:10, 10, 0.2)

par(mfrow=c(1,3))
## 100
set.seed(101)
nn <- 100
sampvec <- rbinom(nn,10,0.2)
plot(table(sampvec)/sum(table(sampvec)), xlim=c(0,10), ylim=c(0,0.35), ylab="",
     main="sample dimension 100")
points(0:10, dbinom(0:10,10,0.2), col=2, lwd=2, cex=0.3)
## 1000
set.seed(101)
nn <- 1000
sampvec <- rbinom(nn,10,0.2)
plot(table(sampvec)/sum(table(sampvec)), xlim=c(0,10), ylim=c(0,0.35), ylab="",
     main="sample dimension 1000")
points(0:10, dbinom(0:10,10,0.2), col=2, lwd=2, cex=0.3)
## 10000
set.seed(101)
nn <- 10000
sampvec <- rbinom(nn,10,0.2)
plot(table(sampvec)/sum(table(sampvec)), xlim=c(0,10), ylim=c(0,0.35), ylab="",
     main="sample dimension 10000")
points(0:10, dbinom(0:10,10,0.2), col=2, lwd=2, cex=0.3)
par(mfrow=c(1,1))

## Poisson distribution

par(mfrow=c(2,2))
xx<-seq(0,30,1)
plot(xx,dpois(xx,0.5),pch=19,ylim=c(0,0.7),
     cex.axis=1.5,xlab=" ",ylab=" ",main="lambda=0.5")
segments(0,0,30,0,lwd=2)
plot(xx,dpois(xx,2),pch=19,ylim=c(0,0.7),
     cex.axis=1.5,xlab=" ",ylab=" ",main="lambda=2")
segments(0,0,30,0,lwd=2)
plot(xx,dpois(xx,5),pch=19,ylim=c(0,0.7),
     cex.axis=1.5,xlab=" ",ylab=" ",main="lambda=5")
segments(0,0,30,0,lwd=2)
plot(xx,dpois(xx,15),pch=19,ylim=c(0,0.7),
     cex.axis=1.5,xlab=" ",ylab=" ",main="lambda=15")
segments(0,0,30,0,lwd=2)
par(mfrow=c(1,1))

## Geometric distribution

par(mfrow=c(2,1))
xx<-seq(1,15,1)
plot(xx,dgeom(xx-1,0.25),pch=19,ylim=c(0,0.6),
     cex.axis=1.5,xlab=" ",ylab=" ",main="p=0.25")
segments(0,0,15,0,lwd=2)
plot(xx,dgeom(xx-1,0.5),pch=19,ylim=c(0,0.6),
     cex.axis=1.5,xlab=" ",ylab=" ",main="p=0.5")
segments(0,0,15,0,lwd=2)
par(mfrow=c(1,1))

## Continuous uniform distribution

xx<-seq(0,1,0.01)
plot(xx,dunif(xx,0,1),xlim=c(-0.5,1.5),ylim=c(0,1.5),
     type='l',lwd=2,cex.axis=1.5,xlab="x",ylab="f(x)")
segments(-0.5,0,0,0,lwd=2)
segments(1,0,1.5,0,lwd=2)

## Exponential distribution

par(mfrow=c(2,2))
lam <- 0.5
xx<-seq(0,5/lam,0.01/lam)
plot(xx,dexp(xx,lam),xlim=c(0,5),
     type='l',lwd=2,cex.axis=1.5,xlab=" ",ylab=" ", main="lambda = .5")
points(0,dexp(0,lam),pch=19,lwd=2)
segments(-1,0,0,0,lwd=2)
lam <- 1
xx<-seq(0,5/lam,0.01/lam)
plot(xx,dexp(xx,lam),xlim=c(0,5/lam),
     type='l',lwd=2,cex.axis=1.5,xlab=" ",ylab=" ", main="lambda = 1")
points(0,dexp(0,lam),pch=19,lwd=2)
segments(-1,0,0,0,lwd=2)
lam <- 2
xx<-seq(0,5/lam,0.01/lam)
plot(xx,dexp(xx,lam),xlim=c(0,5/lam),
     type='l',lwd=2,cex.axis=1.5,xlab=" ",ylab=" ", main="lambda = 2")
points(0,dexp(0,lam),pch=19,lwd=2)
segments(-1,0,0,0,lwd=2)
lam <- 5
xx<-seq(0,5/lam,0.01/lam)
plot(xx,dexp(xx,lam),xlim=c(0,5/lam),
     type='l',lwd=2,cex.axis=1.5,xlab=" ",ylab=" ", main="lambda = 5")
points(0,dexp(0,lam),pch=19,lwd=2)
segments(-1,0,0,0,lwd=2)
par(mfrow=c(1,1))

## Normal distribution

xx<-seq(-4.5,4.5,0.01)
plot(xx,dnorm(xx,0,1),ylim=c(0,0.6),
     type='l',lwd=2,cex.axis=1.3,xlab="x",ylab="f(x)") # Step 1
lines(xx,dnorm(xx,1,1),lwd=2,lty=2) # Step 2
lines(xx,dnorm(xx,0,sqrt(2)),lwd=2,lty=3) # Step 3
lines(xx,dnorm(xx,0,sqrt(1/2)),lwd=2,lty=4) # Step 4

xx<-seq(-4,4,0.01)
plot(xx,dnorm(xx,0,1),ylim=c(0,0.45),type='l',lwd=2,cex.axis=1.3,xlab=" ",ylab=" ")
cord.x <- c(-4,seq(-4,-1.7,0.01),-1.7) 
cord.y <- c(0,dnorm(seq(-4,-1.7,0.01)),0) 
polygon(cord.x,cord.y,col='skyblue')
cord.x <- c(1.7,seq(1.7,4,0.01),4) 
cord.y <- c(0,dnorm(seq(1.7,4,0.01)),0) 
polygon(cord.x,cord.y,col='skyblue')
abline(0,0)

p <- 0.2
qnorm(p, 0, 1)
qnorm(1-p, 0, 1)
z <- 1.96
pnorm(z, 0, 1)
1 - pnorm(-z, 0, 1)
z <- 1.96
dnorm(z, 0, 1)
dnorm(-z, 0, 1)

## The central limit theorem

par(mfrow=c(2,2))
xx<-seq(0,8,1)
plot(xx,dpois(xx,0.5),pch=19,lwd=2,xlim=c(-3,8),cex.axis=1.5,
     xlab=" ",ylab=" ",main="n=1")
curve(dnorm(x,0.5,sqrt(0.5)),-3,8,lwd=2,col='red',add=T)

xx<-seq(0,12,1)
plot(xx,dpois(xx,0.5*5),pch=19,lwd=2,xlim=c(-3,12),cex.axis=1.5,
     xlab=" ",ylab=" ",main="n=5")
curve(dnorm(x,0.5*5,sqrt(0.5*5)),-3,12,lwd=2,col='red',add=T)

xx<-seq(0,30,1)
plot(xx,dpois(xx,0.5*30),pch=19,lwd=2,xlim=c(0,30),cex.axis=1.5,
     xlab=" ",ylab=" ",main="n=30")
curve(dnorm(x,0.5*30,sqrt(0.5*30)),0,30,lwd=2,col='red',add=T)

xx<-seq(20,80,1)
plot(xx,dpois(xx,0.5*100),pch=19,lwd=2,xlim=c(20,80),cex.axis=1.5,
     xlab=" ",ylab=" ",main="n=100")
curve(dnorm(x,0.5*100,sqrt(0.5*100)),20,80,lwd=2,col='red',add=T)
par(mfrow=c(1,1))

par(mfrow=c(2,2))
n <- 5
p <- 0.2
plot(0:n,dbinom(0:n,n,p),pch=19,lwd=2,xlim=c(-2,n),ylim=c(0,0.45),cex.axis=1.5,
     xlab=" ",ylab=" ",main="p=0.2 and n=5")
curve(dnorm(x,n*p,sqrt(n*p*(1-p))),-2,n,lwd=2,col='red',add=T)
n <- 10
plot(0:n,dbinom(0:n,n,p),pch=19,lwd=2,xlim=c(-2,n),ylim=c(0,0.35),cex.axis=1.5,
     xlab=" ",ylab=" ",main="p=0.2 and n=10")
curve(dnorm(x,n*p,sqrt(n*p*(1-p))),-2,n,lwd=2,col='red',add=T)
n <- 30
plot(0:n,dbinom(0:n,n,p),pch=19,lwd=2,xlim=c(-2,n-10),cex.axis=1.5,
     xlab=" ",ylab=" ",main="p=0.2 and n=30")
curve(dnorm(x,n*p,sqrt(n*p*(1-p))),-2,n,lwd=2,col='red',add=T)
n <- 50
plot(0:n,dbinom(0:n,n,p),pch=19,lwd=2,xlim=c(-2,n-20),cex.axis=1.5,
     xlab=" ",ylab=" ",main="p=0.2 and n=50")
curve(dnorm(x,n*p,sqrt(n*p*(1-p))),-2,n,lwd=2,col='red',add=T)
par(mfrow=c(1,1))

par(mfrow=c(1,2))
n <- 5
p <- 0.5
plot(0:n,dbinom(0:n,n,p),pch=19,lwd=2,ylim=c(0,0.4),xlim=c(-2,n+2),
     cex.axis=1.5,xlab=" ",ylab=" ",main="p=0.5 and n=5")
curve(dnorm(x,n*p,sqrt(n*p*(1-p))),-2,n+2,lwd=2,col='red',add=T)
n <- 10
plot(0:n,dbinom(0:n,n,p),pch=19,lwd=2,xlim=c(-2,n+2),
     cex.axis=1.5,xlab=" ",ylab=" ",main="p=0.5 and n=10")
curve(dnorm(x,n*p,sqrt(n*p*(1-p))),-2,n+2,lwd=2,col='red',add=T)

set.seed(25)
xx <- c(rnorm(10000000,10,3),rnorm(10000000,40,9)) # Step 1
hist(xx)

par(mfrow=c(2,2)) # Step 2

nsamples <- 1000 # Step 3

sampmean <- NULL # Step 4
size <- 2 
set.seed(21)
for(i in 1:nsamples){sampmean <- c(sampmean, 
                                   mean(sample(xx,size,replace=T)))} # Step 5
plot(density(sampmean), main="n=2") # Step 6
curve(dnorm(x,mean(xx),sqrt(var(xx)/size)),lwd=2,col='red',add=T)

sampmean <- NULL # Step 4
size <- 5
set.seed(25)
for(i in 1:nsamples){sampmean <- c(sampmean, 
                                   mean(sample(xx,size,replace=T)))} # Step 5
plot(density(sampmean),ylim=c(0,0.06), main="n=5") # Step 6
curve(dnorm(x,mean(xx),sqrt(var(xx)/size)),lwd=2,col='red',add=T)

sampmean <- NULL # Step 4
size <- 15 
set.seed(22)
for(i in 1:nsamples){sampmean <- c(sampmean, 
                                   mean(sample(xx,size,replace=T)))} # Step 5
plot(density(sampmean),ylim=c(0,0.1), main="n=15") # Step 6
curve(dnorm(x,mean(xx),sqrt(var(xx)/size)),lwd=2,col='red',add=T)

sampmean <- NULL # Step 4
size <- 40 
set.seed(25)
for(i in 1:nsamples){sampmean <- c(sampmean, 
                                   mean(sample(xx,size,replace=T)))} # Step 5
plot(density(sampmean), main="n=40") # Step 6
curve(dnorm(x,mean(xx),sqrt(var(xx)/size)),lwd=2,col='red',add=T)
par(mfrow=c(1,1))

## Chi-squared distribution

xx<-seq(0.01,25,0.01)
plot(xx,dchisq(xx,1),ylim=c(0,0.4),xlim=c(0,25),type='l',lwd=2,
     cex.axis=1.3,xlab="x",ylab="f(x)")
lines(xx,dchisq(xx,3),lwd=2,lty=2)
lines(xx,dchisq(xx,6),lwd=2,lty=3)
lines(xx,dchisq(xx,10),lwd=2,lty=4)
legend(15,0.3,legend=c("1 d.f.","3 d.f.","6 d.f.","10 d.f."), lty=1:4, lwd=2)
par(mfrow=c(1,1))

## Student's t distribution

xx<-seq(-5,5,0.01)
plot(xx,dnorm(xx),type='l',ylim=c(0,0.45),lwd=2,cex.axis=1.3,xlab="x",
     ylab="f(x)", col='red')
lines(xx,dt(xx,1),lwd=2)
lines(xx,dt(xx,3),lwd=2,lty=2)
lines(xx,dt(xx,50),lwd=3,lty=3)
legend(2,.4,legend=c("Standard Gaussian","Student's t - 1 d.f.",
                     "Student's t - 3 d.f.","Student's t - 50 d.f."), 
                     lty=c(1,1,2,3), col=c(2,1,1,1))

## F distribution

xx<-seq(0,6,0.01)
plot(xx,df(xx,5,5),type='l',ylim=c(0,1.25),lwd=2,
     cex.axis=1.3,xlab="x",ylab="f(x)",
     main=" ")
lines(xx,df(xx,5,25),lwd=2,lty=2)
lines(xx,df(xx,25,5),lwd=2,lty=3)
lines(xx,df(xx,25,25),lwd=2,lty=4)
legend(4,1,legend=c("k=5,m=5","k=5,m=25","k=25,m=5","k=25,m=25"), lty=1:4)


### Simulation of random samples ###

## Simulations from a normal distribution

par(mfrow=c(1,3),pty="s")
x <- seq(46,56,0.05)
mean(nhtemp)
var(nhtemp)
hist(nhtemp,freq=F,xlim=c(46,56),ylim=c(0,0.45),main=' ',xlab=' ',ylab=' ')
lines(density(nhtemp),lwd=2)
lines(x,dnorm(x,mean(nhtemp),sqrt(var(nhtemp))),col='red',lwd=2)
set.seed(123)
for(i in 1:2){
  y <- rnorm(60,mean(nhtemp),sqrt(var(nhtemp)))
  hist(y,freq=F,main=' ',xlim=c(46,56),ylim=c(0,0.45),xlab=' ',ylab=' ')
  lines(density(y),lwd=2)
  lines(x,dnorm(x,mean(nhtemp),sqrt(var(nhtemp))),col='red',lwd=2)
}
par(mfrow=c(1,1))

## Simulation of the bernoulli sample mean

set.seed(4)
x <- seq(0,1.5,0.01)
sim1<-rbinom(1000,25,0.25)/25 # 1000 simulated sample means with n=25
sim2<-rbinom(1000,50,0.25)/50 # 1000 simulated sample means with n=50
sim3<-rbinom(1000,100,0.25)/100 # 1000 simulated sample means with n=100

par(mfrow=c(1,3),pty="s")
hist(sim1,freq=F,xlab="n=25",ylab=' ',main=' ')
lines(x,dnorm(x,0.25,sqrt(0.25*0.75/10)),lwd=2,col='red')
lines(density(sim1),lwd=2)
hist(sim2,freq=F,xlab="n=50",ylab=' ',main=' ')
lines(x,dnorm(x,0.25,sqrt(0.25*0.75/50)),lwd=2,col='red')
lines(density(sim2),lwd=2)
hist(sim3,freq=F,xlab="n=100",ylab=' ', main=' ')
lines(x,dnorm(x,0.25,sqrt(0.25*0.75/100)),lwd=2,col='red')
lines(density(sim3),lwd=2)
par(mfrow=c(1,1))

# Simulation of a bernoulli variance estimation

set.seed(4)
sim<-rbinom(1000,100,0.25)/100 # 1000 simulated sample means with n=100
varest <- sim*(1-sim)
hist(varest,freq=F,xlab=" ",ylab=' ',main=' ')
lines(density(varest),lwd=2)

## Simulation of regression data

plot(depression ~ weight, data = roller, xlim=c(0,1.04*max(weight)), # Step 1
     ylim=c(0,1.04*max(depression)), xlab = 'Weight of roller', 
     ylab = 'Depression', pch = 16)

roller.lm <- lm(depression ~ weight,data=roller) # Step 2
abline(roller.lm,col='red',lwd=2)

roller.sim <- simulate(roller.lm,nsim=20) # Step 3
for(i in 1:20) { # Step 4
abline(lm(roller.sim[,i]~ roller$weight),lty=2)
}


### Model assumptions ###

## Example: onions data

onions<-read.table("onions.dat", col.names=c("yield","dens","location")) # Step 1
par(mfrow=c(2,2))

hist(onions$yield,freq=F,main=' ',ylim=c(0,0.016),xlim=c(0,200),ylab=' ') # Step 2a
curve(dnorm(x,mean(onions$yield),sqrt(var(onions$yield))),0,200,add=T,lwd=2,col='red')
qqnorm(onions$yield,main=' ',xlab=' ',ylab=' ') # Step 2b
qqline(onions$yield,col='red',lwd=2)

hist(onions$dens,freq=F,main=' ',ylim=c(0,0.010),xlim=c(0,310),ylab=' ') # Step 3a
curve(dnorm(x,mean(onions$dens),sqrt(var(onions$dens))),0,310,add=T,lwd=2,col='red')
qqnorm(onions$dens,main=' ',xlab=' ',ylab=' ') # Step 3b
qqline(onions$dens,col='red',lwd=2)
par(mfrow=c(1,1))

ks.test(onions$yield,pnorm)
ks.test(onions$dens,pnorm)

library(fBasics)
ksnormTest(onions$yield)
ksnormTest(onions$dens)

shapiroTest(onions$yield)
shapiroTest(onions$dens)

jarqueberaTest(onions$yield)
jarqueberaTest(onions$dens)

dagoTest(onions$yield)
dagoTest(onions$dens)

library(nortest)
# adTest(onions$yield)
# adTest(onions$dens)

cvmTest(onions$yield)
cvmTest(onions$dens)

lillieTest(onions$yield)
lillieTest(onions$dens)

pchiTest(onions$yield)
pchiTest(onions$dens)

sfTest(onions$yield)
sfTest(onions$dens)


onions<-read.table("onions.dat", col.names=c("yield","dens","location"))
logyield<-log(onions$yield)
logdens<-log(onions$dens)
par(mfrow=c(2,2))
hist(logyield,freq=F,main=' ',ylim=c(0,0.7),xlim=c(2.5,5.5),ylab=' ')
curve(dnorm(x,mean(logyield),sqrt(var(logyield))),2.5,5.5,add=T,lwd=2,col='red')
qqnorm(logyield,main=' ',xlab=' ',ylab=' ')
qqline(logyield,col='red',lwd=2)
hist(logdens,freq=F,main=' ',ylim=c(0,1),xlim=c(3,6),ylab=' ')
curve(dnorm(x,mean(logdens),sqrt(var(logdens))),3,6,add=T,lwd=2,col='red')
qqnorm(logdens,main=' ',xlab=' ',ylab=' ')
qqline(logdens,col='red',lwd=2)
par(mfrow=c(1,1)) 

dagoTest(logyield)
dagoTest(logdens)

