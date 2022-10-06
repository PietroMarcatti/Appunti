######### Lab 7 - Unsupervised methods #############


## Install the packages "MASS", "cluster"  ##

# install.packages("MASS")
# install.packages("cluster")


### Example: US arrest data ###

str(USArrests)
summary(USArrests)
pairs(USArrests, panel = panel.smooth, pch=16, lwd=2)

obj <- princomp(USArrests, cor=TRUE) 
z1 <- -obj$scores[,1] # the sign of the scores is modified
boxplot(cbind(scale(USArrests), z1), col=c(rep(5,4),2))

phi1<--obj$loadings[,1] # the sign of the loadings is modified
phi1 # 1st principal component loadings
phi2<--obj$loadings[,2] # the sign of the loadings is modified
phi2 # 2nd principal component loadings

obj$loadings<--obj$loadings # the sign of the loadings is modified
obj$scores<--obj$scores # the sign of the scores is modified
biplot(obj, xlab="1st principal component", ylab="2nd principal component", 
       xlim=c(-3.5,3.5), col=c(1,2), scale=0)

par(mfrow=c(1,2), pty="s")
plot(obj$sdev^2/4, xlab="Principal component", ylab="PVE", type='b')
plot(cumsum(obj$sdev^2)/4, xlab="Principal component", ylab="Cumulative PVE", 
     ylim=c(0,1), type='b')
par(mfrow=c(1,1))


### Example: three clusters simulated data ###

set.seed(25)
x<-matrix(rnorm(75*2), ncol=2)
x[1:25,1]<-x[1:25,1]+2
x[1:25,2]<-x[1:25,2]-2
x[26:50,1]<-x[26:50,1]+3
plot(x[1:25,1], x[1:25,2], xlim=c(-2,4.5), ylim=c(-4,2), type='n', xlab='X1',
     ylab='X2')
text(x[1:25,1], x[1:25,2])
text(x[26:50,1], x[26:50,2],labels = seq(26,50,1),col='red')
text(x[51:75,1], x[51:75,2],labels = seq(51,75,1),col='blue')

hc.complete<-hclust(dist(x), method="complete")
plot(hc.complete, xlab="", sub="", cex=.9) 
abline(5,0,lty=2)
# plot(hc.complete, hang=-1, xlab="", sub="", cex=.9) 
# abline(5,0,lty=2)

cutree(hc.complete, 3)

hc.average<-hclust(dist(x), method="average")
plot(hc.average, xlab="", sub="", cex=.9) 
abline(2.6,0,lty=2)
cutree(hc.average, 3)

hc.centroid<-hclust(dist(x), method="centroid")
plot(hc.centroid, xlab="", sub="", cex=.9) 
abline(1.5,0,lty=2)
cutree(hc.centroid, 3)

set.seed(5)
km3<-kmeans(x, 3, nstart =20)
km3

plot(x[1:25,1], x[1:25,2], xlim=c(-2,4.5), ylim=c(-4,2), type='n', 
     xlab='X1', ylab='X2')
text(x[which(km3$cluster==1),1], x[which(km3$cluster==1),2],
     labels = which(km3$cluster==1))
text(x[which(km3$cluster==2),1], x[which(km3$cluster==2),2],
     labels = which(km3$cluster==2),col='red')
text(x[which(km3$cluster==3),1], x[which(km3$cluster==3),2],
     labels = which(km3$cluster==3),col='blue')
points(km3$centers[1,1], km3$centers[1,2], pch=19, cex=1.5)
points(km3$centers[2,1], km3$centers[2,2], pch=19, col='red', cex=1.3)
points(km3$centers[3,1], km3$centers[3,2], pch=19, col='blue', cex=1.3)

library(cluster)
me3<-pam(x, 3)
me3

plot(x[1:25,1], x[1:25,2], xlim=c(-2,4.5), ylim=c(-4,2),type='n', 
     xlab='X1', ylab='X2')
text(x[which(me3$cluster==1),1], x[which(me3$cluster==1),2],
     labels = which(me3$cluster==1))
text(x[which(me3$cluster==2),1], x[which(me3$cluster==2),2],
     labels = which(me3$cluster==2), col='red')
text(x[which(me3$cluster==3),1], x[which(me3$cluster==3),2],
     labels = which(me3$cluster==3), col='blue')
points(x[1,1], x[1,2], pch=21, cex=3, lwd=2)
points(x[21,1], x[21,2], pch=21, cex=3, col='red', lwd=2)
points(x[74,1], x[74,2], pch=21, cex=3, col='blue', lwd=2)


### Example: Swiss socioeconomic indicators ###

data(swiss)
str(swiss)
swiss.x <- as.matrix(swiss[, -1]) # new data matrix without Fertility
pairs(swiss.x, panel = panel.smooth, pch=16, lwd=2)

h <- hclust(dist(swiss.x), method = "single")
plot(h,cex=.8, xlab=' ',sub=' ', main=' ')
rect.hclust(h, k=3, border='red')

library(cluster)
h1<-diana(swiss.x)
pltree(h1, cex=.8,xlab=' ',sub=' ', main=' ')
rect.hclust(h1, k=3, border='red')

h2 <- hclust(dist(swiss.x), method = "average")
initial <- tapply(swiss.x, list(rep(cutree(h2,3), ncol(swiss.x)), 
                    col(swiss.x)), mean)
dimnames(initial) <- list(NULL, dimnames(swiss.x)[[2]])
initial

h3<-kmeans(swiss.x,centers=initial)
h3

swiss.pca <- prcomp(swiss.x)
swiss.px <- predict(swiss.pca)
swiss.centers <- predict(swiss.pca, h3$centers)
plot(swiss.px[, 1:2], type = "n", xlab = "1st principal component",
     ylab = "2nd principal component")
text(swiss.px[which(h3$cluster==1), 1:2], 
     labels = h3$cluster[which(h3$cluster==1)])
text(swiss.px[which(h3$cluster==2), 1:2], 
     labels = h3$cluster[which(h3$cluster==2)], col='red')
text(swiss.px[which(h3$cluster==3), 1:2], 
     labels = h3$cluster[which(h3$cluster==3)], col='blue')
points(swiss.centers[1,1], swiss.centers[1,2], pch=19, lwd=2)
points(swiss.centers[2,1], swiss.centers[2,2], pch=19, lwd=2, col='red')
points(swiss.centers[3,1], swiss.centers[3,2], pch=19, lwd=2, col='blue')

set.seed(5)
h4<-kmeans(swiss.x,3,nstart=20)
swiss.centers <- predict(swiss.pca, h4$centers)
plot(swiss.px[, 1:2], type = "n", xlab = "1st principal component",
     ylab = "2nd principal component")
text(swiss.px[which(h4$cluster==1), 1:2], 
     labels = h4$cluster[which(h4$cluster==1)])
text(swiss.px[which(h4$cluster==2), 1:2], 
     labels = h4$cluster[which(h4$cluster==2)]+1,col='blue')
text(swiss.px[which(h4$cluster==3), 1:2], 
     labels = h4$cluster[which(h4$cluster==3)]-1, col='red')
points(swiss.centers[1,1], swiss.centers[1,2], pch=19, lwd=2)
points(swiss.centers[2,1], swiss.centers[2,2], pch=19, lwd=2, col='blue')
points(swiss.centers[3,1], swiss.centers[3,2], pch=19, lwd=2, col='red')

h5 <- pam(swiss.x,3)
h5
swiss.medoids <- predict(swiss.pca, h5$medoids)
plot(swiss.px[, 1:2], type = "n", xlab = "1st principal component", 
     ylab = "2nd principal component")
text(swiss.px[which(h5$cluster==3), 1:2], 
     labels = h5$cluster[which(h5$cluster==3)]-2)
text(swiss.px[which(h5$cluster==2), 1:2], 
     labels = h5$cluster[which(h5$cluster==2)], col='red')
text(swiss.px[which(h5$cluster==1), 1:2], 
     labels = h5$cluster[which(h5$cluster==1)]+2, col='blue')
points(swiss.medoids[3,1], swiss.medoids[3,2], pch=21, cex=3, lwd=2)
points(swiss.medoids[2,1], swiss.medoids[2,2], pch=21, cex=3, col='red', lwd=2)
points(swiss.medoids[1,1], swiss.medoids[1,2], pch=21, cex=3, col='blue', lwd=2)

swiss.x <- as.data.frame(swiss.x)
pairs(swiss.x, col = 1 + (swiss.x$Catholic > 50))

