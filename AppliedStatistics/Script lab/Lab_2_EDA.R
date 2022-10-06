######### LAB 2 - Exploratory Data Analysis #############


## Install the packages DAAG, "moments", "ISLR", "vcd", "vioplot"  ##
 
# install.packages("DAAG")
# install.packages("moments")
# install.packages("ISLR")
# install.packages("vcd")
# install.packages("vioplot")


### Tabular and graphical representations ###

## Example: caffeine and marital status

caff.marital <- matrix(c(652,1537,598,242,36,46,38,21,218,327,106,67),nrow=3,byrow=T)
colnames(caff.marital) <- c("0","1-150","151-300",">300")
rownames(caff.marital) <- c("Married","Prev.married","Single")
names(dimnames(caff.marital)) <- c("Marital status","Caffeine consumption in mg/day")

caff.marital

orig.data <- rbind(cbind(rep("0",652), rep("Married",652)),
cbind(rep("1-150",1537), rep("Married",1537)),
cbind(rep("151-300",598), rep("Married",598)),
cbind(rep("300+",242), rep("Married",242)),
cbind(rep("0",36), rep("Prev.married",36)),
cbind(rep("1-150",46), rep("Prev.married",46)),
cbind(rep("151-300",38), rep("Prev.married",38)),
cbind(rep("300+",21), rep("Prev.married",21)),
cbind(rep("0",218), rep("Single",218)),
cbind(rep("1-150",327), rep("Single",327)),
cbind(rep("151-300",106), rep("Single",106)),
cbind(rep("300+",67), rep("Single",67)))
orig.data <- as.data.frame(orig.data)
colnames(orig.data) <- c("marital","consumption")
str(orig.data)

table(orig.data$consumption, orig.data$marital)

total.caff <- margin.table(caff.marital,2)
barplot(total.caff, col="white") # argument col

par(mfrow=c(1,2))
barplot(t(caff.marital),beside=T,legend.text=colnames(caff.marital),col=c("white",
        "grey80","grey50","black"))
barplot(prop.table(t(caff.marital),2),beside=T,col=c("white","grey80","grey50","black"))
par(mfrow=c(1,1))

opar <- par(mfrow=c(1,3),mex=0.8, mar=c(1,1,5,1))
slices <- c("white","grey80","grey50","black")
pie(caff.marital["Married",], main="Married", col=slices)
pie(caff.marital["Prev.married",],
main="Previously married", col=slices)
pie(caff.marital["Single",], main="Single", col=slices)
par(opar)

## Example: possum data set

library(DAAG)
data(possum)
# ?possum
ftotlngth <- with(possum, totlngth[sex=="f"]) 
opar <- par(mfrow = c(2,2), pty="s") 
hist(ftotlngth, breaks = 72.5 + (0:5) * 5, ylim = c(0, 22),xlab="Total length (cm)", 
     main ="A: Breaks at 72.5, 77.5, ...") 
hist(ftotlngth, breaks = 75 + (0:5) * 5, ylim = c(0, 22),xlab="Total length (cm)", 
     main="B: Breaks at 75, 80, ...") 
dens <- density(ftotlngth)  
xlim <- range(dens$x); ylim <- range(dens$y) 
hist(ftotlngth, breaks = 72.5 + (0:5) * 5, probability = T,xlim = xlim, ylim = ylim, 
     xlab="Total length (cm)",main ="C: Breaks as in A") 
lines(dens) 
hist(ftotlngth, breaks = 75 + (0:5) * 5, probability = T,xlim = xlim, ylim = ylim, 
     xlab="Total length (cm)",main="D: Breaks as in B") 
lines(dens) 
par(opar)
par(mfrow=c(1,1))

boxplot(ftotlngth, horizontal=TRUE) 
boxplot(possum$footlgth~possum$sex)


## Example: the milk data set

xyrange <- range(milk) 
par(pty="s")
plot(four ~ one, data = milk, xlim = xyrange, ylim = xyrange, pch = 16)    
# plot(milk$four,milk$one, xlim = xyrange, ylim = xyrange, pch = 16)
rug(milk$one) 
rug(milk$four, side = 2)
abline(0, 1)
par(pty="m")

## Example: electrical resistance of kiwifruit

plot(ohms ~ juice, xlab="Apparent juice content (%)",ylab="Resistance (ohms)", data=fruitohms) 
with(fruitohms, lines(lowess(juice, ohms), lwd=2))
abline(lm(ohms ~ juice,data=fruitohms),col='red',lwd=2)

## Untransformed scale vs logarithmic scale: the animals example

library(MASS)
oldpar <- par(mfrow = c(1,2), pty="s")
plot(brain ~ body, data=Animals)
plot(log(brain) ~ log(body), data=Animals)
par(oldpar)

## Patterns in grouped data: the cuckoos example

cuckoos$specnam <- with(cuckoos, sub(pattern=".", replacement=" ",species, fixed=TRUE)) 
plt1 <- stripplot(specnam ~ length, data=cuckoos)

#specnam <- with(cuckoos, sub(pattern=".", replacement=" ",levels(species), fixed=TRUE)) 
#plt1 <- stripplot(species ~ length, factor.levels="specnam", data=cuckoos)

print(update(plt1, xlab="Length of egg (mm)"),position=c(0,0,0.55,1))  # xmin, ymin, xmax, ymax 
plt2 <- bwplot(specnam ~ length, xlab="Length of egg (mm)",scales=list(y=list(alternating=0)), 
        data=cuckoos) 
print(plt2,newpage=FALSE, position=c(0.55,0,1,1))

## Comparing several time series: the labor force example

xyplot(Ontario+Quebec+BC+Alberta+Prairies+Atlantic ~ Date,outer=FALSE, data=jobs, type="b", 
       ylab="Number of workers",auto.key=list(space="right", lines=TRUE))

xyplot(Ontario+Quebec+BC+Alberta+Prairies+Atlantic ~ Date,data=jobs, type="b", layout=c(3,2), 
       ylab="log(Number of workers)",scales=list(y=list(relation="sliced", log=TRUE)),outer=TRUE)

## Comparing several scatterplots: the tinting example

tint.xyplot <-xyplot(csoa ~ it|sex*agegp, groups=tint,data=tinting, aspect=1,
      type=c("p","smooth"),span=1.25)
update(tint.xyplot, legend=NULL, auto.key=list(columns=3,points=TRUE, lines=TRUE))


### Numerical summaries ###

## Measures of location: the possum example

summary(possum$footlgth)

min(possum$footlgth, na.rm=T) # that is also quantile(possum$footlgth,0, na.rm=T)
quantile(possum$footlgth,0.25, na.rm=T)
median(possum$footlgth, na.rm=T) # that is also quantile(possum$footlgth,0.5, na.rm=T)
mean(possum$footlgth, na.rm=T)
quantile(possum$footlgth,0.75, na.rm=T)
max(possum$footlgth, na.rm=T) # that is also quantile(possum$footlgth,1, na.rm=T)

## Measures of variability: the possum example

var(possum$footlgth, na.rm=T) # the corrected sample variance
sd(possum$footlgth, na.rm=T)
range(possum$footlgth, na.rm=T)
IQR(possum$footlgth, na.rm=T) # the difference between the third and the first quantiles

cond <- !is.na(possum$footlgth)
numb <- length(possum$footlgth[cond])
sqrt(sum((possum$footlgth[cond]-sum(possum$footlgth[cond])/numb)^2)/(numb-1))

var(possum$footlgth, na.rm=T)*((numb-1)/numb)

mean((possum$footlgth - mean(possum$footlgth, na.rm=T))^2, na.rm=T)

mean(possum$footlgth^2, na.rm=T) - mean(possum$footlgth, na.rm=T)^2

## Measures of skewness and kurtosis: the wage example

library(moments)
library(ISLR)

attach(Wage)
par(mfrow=c(2,1))
hist(wage)
abline(v=mean(wage), col=2)
abline(v=median(wage), col=3)
mean(wage)
median(wage)
hist(logwage)
abline(v=mean(logwage), col=2)
abline(v=median(logwage), col=3)
mean(logwage)
median(logwage)
par(mfrow=c(1,1))

skewness(wage)
skewness(logwage)

kurtosis(wage)
kurtosis(logwage)

detach(Wage)

### Multivariate exploratory data analysis ###

## Correlation analysis

set.seed(104)
x <- (11:30)/5
y <- 2 + 0.15 * x + 0.6 * rnorm(20)
plot(x,y,pch=19)
cor(x,y)
abline(lm(y~x),col="red")

set.seed(105)
x <- (11:30)/5
y <- 2 + 0.15 * x + 0.2 * rnorm(20)
plot(x,y,pch=19)
cor(x,y)
abline(lm(y~x),col="red")

set.seed(106)
x <- (11:30)/5
y <- 2 + 0.15 * x + 0.06 * rnorm(20)
plot(x,y,pch=19)
cor(x,y)
abline(lm(y~x),col="red")

## Non-linearity in correlation analysis

set.seed(103)
x <- (11:30)/5
y <- 2 - 0.05 * x + 0.1 * ((x - 1.75))^4 + 1.25 * rnorm(20)
plot(x,y,pch=19)
abline(lm(y~x),col="red")

cor(x,y,method="pearson")
cor(x,y,method="spearman")

## The relationship between two qualitative variables

SalaryCl <- cut(ISLR::Hitters$Salary, c(0,200,500,700,2500))
contTable <- table(ISLR::Hitters$League, SalaryCl)
chisq.test(contTable)

## The caffeine consumption example

chisq.test(caff.marital)


### Multiple phenomena representation: some further examples ###

## The mosaic plot

library(vcd)
stones <- array(c(81,6,234,36,192,71,55,25), dim=c(2,2,2),
                dimnames=list(Sucess=c("yes","no"),
Method=c("open","ultrasound"),Size=c("<2cm\n", ">=2cm\n")))
stones
mosaic(aperm(stones, 3:1), main=NULL,labeling_args=list(gp_labels=gpar(fontsize=12),
        gp_varnames=gpar(fontsize=12)),legend_args=list(fontsize=12))

## The barplot

attach(Wage)
par(mfrow=c(1,1))
table(education, jobclass) -> edu.jclass
barplot(t(edu.jclass),2,beside=T,legend.text=colnames(edu.jclass),col=c("white","black"))
barplot(prop.table(t(edu.jclass),2),beside=T,col=c("white","black"))
detach(Wage)

## The violin plot

library(vioplot)
library(ISLR)
par(mfrow=c(1,2))
boxplot(ISLR::Hitters$Salary~ISLR::Hitters$League)
vioplot(ISLR::Hitters$Salary[ISLR::Hitters$League=="A"],ISLR::Hitters$Salary[ISLR::Hitters$League=="N"]) 

# there is an issue with NAs
cond <- !is.na(ISLR::Hitters$Salary)
vioplot(ISLR::Hitters$Salary[cond][ISLR::Hitters$League[cond]=="A"],
        ISLR::Hitters$Salary[cond][ISLR::Hitters$League[cond]=="N"],names=c("A","N"), col="grey")

## Conditional measures of location and variability

tapply(ISLR::Hitters$Salary, ISLR::Hitters$League, summary)
tapply(ISLR::Hitters$Salary[cond], ISLR::Hitters$League[cond], var)
tapply(ISLR::Hitters$Salary[cond], ISLR::Hitters$League[cond], sd)
