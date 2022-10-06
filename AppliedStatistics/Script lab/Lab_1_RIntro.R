######### LAB 1 - R Intro ############

### Using R ###

# q(), or use the graphical menu, to quit R.

# savehistory("history.r")

# loadhistory("history.r")

# source("command.r")

# save.image("myfile.Rdata")

# load("myfile.RData") 

# R gets confused if you use a path in your code like c:\mydocuments\myfile.txt
# This is because R sees "\" as an escape character. Instead, use c:/mydocuments/myfile.txt
# Use the slash symbol instead of the backslash or the double backslash

getwd()   # print the current working directory
# setwd(c:/docs/mydirectory)  # change the current working directory to "mydirectory"
ls()   # list the objects in the current workspace

### R as calculator ###

# + and - for addition and subtraction
# * and / for multiplication and division
# ^ for exponents
# %% is the modulo operator
# %\% for integer division

# print() # prints objects
# log() # computes logarithms
# exp() # computes the exponential function
# sqrt() # takes the square root
# abs() # returns the absolute value
# sin() # returns the sine
# cos() # returns the cosine
# tan() # returns the tangent
# asin() # returns the arc-sine
# factorial() # returns the factorial
# sign() # returns the sign (negative or positive)
# round() # rounds the input to the desired digit

#  # this is a comment
1+2+3
2+3*4
3/2+1
2+(3*4)
(2 + 3) * 4
4*3^3
27^(1/3)
2/0 # the result is (positive) infinity
0/0 # the result is Not a Number (NaN)
23%%3
23%/%3

sqrt(2)
sin(3.14159)
sin(pi)

### Logical Operators ###

# < less than
# <= less than or equal to
# > greater than
# >= greater than or equal to
# == equal
# != not equal
# & and
# | or

1 == 1
1 == 2
1 != 2
1 <= 2 & 1 <= 3
1 == 1 | 1 == 2
1 > 1 | 1 > 2 & 3 == 3
1 > 1 & 1 > 2 & 1 > 3
xor(TRUE, TRUE)
xor(TRUE, FALSE)

### R's help function ###

?log
help(log)
apropos("mean") # when the name is not known exactly

### Packages and libraries ###

# install.packages("package")  to istall a package

# library("package")  to load the package

# update.packages()  To update packages


### Assignment and reference ###

1 + 2 # the result is not saved, it is only printed on the screen 
x <- sqrt(2)   # the square root of 2 is saved in x
x   # in order to print the content of x
x^3
y <- x^3
y
x <- pi # a new assignment erases the former one
x
is(x)

ls() # to see what objects are in the current working directory
rm(x) # to remove object x
# rm(list = ls()) # to remove all objects

### Vectors ###

Vector1 <- c(1,2,3,4,5,6,7,8,9,10) # numerical vector
Vector1
x <- c(2,3,5,7)
x
x <- c(x,11)
x

Vector2 <- c("a","b","c","d") # character vector
Vector2
Vector3 <- c("1","2","3","4")  # character vector
Vector3

x = c(TRUE, FALSE, TRUE, FALSE)
y = !x
x
y
x & y
x | y

Vector4 <- c(Vector2 , Vector3 , Vector2 , Vector2 , Vector2)
Vector4

xx <- 1:10
xx
5:-5

seq(from=0,to=10) # you can drop the argument names
seq (0,10)
seq (0,10,by = 2) # the by argument let's you set the increments
seq (0,10,length.out=25) # the length.out argument specifies the length of the vector and R figures out the increments itself

rep(0,time=10)
rep("Hello",3) # you can drop the argument name
rep(Vector1,2)
rep(Vector2,each=2) # we can repeat each element as well

### Vector operations ###

# applied element-wise. 

# sum()   sums of the elements of the vector
# prod()   product of the elements of the vector
# min()   minimum of the elements of the vector
# max()   maximum of the elements of the vector
# mean()   mean of the elements
# median()   median of the elements
# range()  the range of the vector
# sd()   the standard deviation
# var()   the variance (on n-1)
# cov()   the covariance (takes two inputs cov(x,y))
# cor()   the correlation coefficient (takes two imputs cor(x,y))
# sort()    sorts the vector (default argument: decreasing = FALSE)
# order()   indices of the elements in ascending order
# length()   returns the length of the vector
# summary()   returns summary statistics
# which()   returns the index after evaluating a logical statement
# which.min()    returns the index of the min
# which.max()   returns the index of the max
# unique()   returns a vector of all the unique elements of the input
# round()    rounds the values in its first argument to the specified number of decimal places (the default is 0)

x <- 0:10
x+1
y <- -5:5
abs(y)
x+y
x*y
y <- 0:8
x+y # warning message, when two objects have different lengths, R reuses elements of the shorter one

x
x > 5
x <- 3:26
max(x)
min(x)
sum(x)
prod(x)
x <- c(32,18,25:21,40,17)
x
sort(x)  # ascending order
order(x) # position indices of the elements in ascending order

Vector1+Vector1
Vector1/Vector1
log(Vector1)
round(log(Vector1))
round(log(Vector1),digit = 3)
which(Vector1>=5) # this returns the indices not the elements


### Vector indexing ###

Vector6 <- c ("The","Starlab","Fellow","is","a Fool")
Vector6[3]
Vector6[2:4]
Vector6[c(1 ,3 ,4)]
Vector6[-2] # all except the 2nd element
Vector6[5] <- "great"
Vector6

xx <- 100:1
xx[7]
xx[c(2,3,5,7,11)]
xx[85:91]
xx[91:85]
xx[c(1:5,8:10)]
xx[c(1,1,1,1,2,2,2,2)]
yy <- xx[c(1,2,4,8,16,32,64)]
yy
x <- c(32,18,25:21,40,17)
x
sort(x)
x[order(x)]
x <- c(1,2,4,8,16,32)
x
x[-4]
x[-c(3,4)]

x <- -8:7
x<0
x[x<0]
x[x<0&x<(-2)]
x[x<0|x>5]
x[x!=6]
x[x==6]

### More functions ###

z<-c(2,3,4,3,NA,NA,6,6,10,11,2,NA,4,3)
max(z) # this won't work because many function can't deal with NAs
na.omit(z) # returns the vector supressing the Nas
max(na.omit(z))
max(z,na.rm=T)

is.na(z)

z.noNA <- subset(z,is.na(z)==FALSE )
z.noNA

X <- 1:70
Multiple7 <- subset(X,X%%7==0) # recall the modulo operator
Multiple7

### Ordered and unordered factors ###

treat <- factor(c("a","b","b","c","a","b"))
treat
levels(treat)

treat1 <- ordered(c("a","b","b","c","a","b"), levels=c("c","b","a")) # to set a different ordering
treat1
levels(treat1)

resp <- c(10,3,7,6,4,5)
resp[treat=="a"]
sum(resp[treat=="b"])

x<-c(1:12,25:38,-3:0,13:24)
x1 <- cut(x,c(-5,5,25,40),labels=c("B","M","A")) # classes: (-5,5],(5,25],(25,40] corresponding to B, M, A
x1
levels(x1)
f_x1 <- table(x1) # frequencies for each level
f_x1

x<-1:20
y<-factor(rep(0:1,10))
y
tapply(x,y,sum) # apply sum to each group of values in x given by a unique combination of the levels of the factors y

### Arrays and matrices ###

Matrix1 <- matrix(data=1,nrow=3,ncol=3)
Matrix1
dim(Matrix1)

Vector8 <- 1:12
Vector8
Matrix3 <- matrix(data=Vector8,nrow=4)  # by default the matrix will be populated by column
Matrix3
Matrix4 <- matrix(data=Vector8,nrow=4,byrow=TRUE)  # now we populated it by row
Matrix4

Vector9 <- 1:10
Vector9
Vector10 <- Vector9^2
Vector10
Matrix5 <- rbind(Vector9,Vector10)
Matrix5
Matrix6 <- cbind(Vector9,Vector10,Vector9)
Matrix6
colnames(Matrix6)
rownames(Matrix6) # the rownames do not exist
colnames(Matrix6) <- c("A","B","C")
rownames(Matrix6) <- c("a","b","c","d","e","f","g","h","i","j")
Matrix6

Matrix7 <- diag(5) # creates a 5 by 5 identity matrix
Vector11 <- c(1,2,3,4,5)
Matrix8 <- diag(Vector11) # Vector11 across the diagonal
diag(Matrix7) # extracts the diagonal from a matrix

### Mathematical operations ###

# + - * / standard scalar or by element operations
# %*% matrix multiplication
# t() transpose
# solve() inverse
# det() determinant
# chol() cholesky decomposition
# eigen() eigenvalues and eigenvectors
# crossprod() cross product
# %x% kronecker product

### Indexing matrices ###

Matrix9 <- matrix(1:9,3)
Matrix9
Matrix9[1,1] # extracts the first element of the first row
Matrix9[2,3] # extracts the third element of the second row
Matrix9[,1] # extracts the first column
Matrix9[2,] # extracts the second row
Matrix9[1:2 , ] # extracts the first and second row
Matrix9[Matrix9[,2]>4,] # extracts all rows that in their second column contain values greater than four

x <- matrix(1:16,ncol=4)
x
apply(x,1,sum) # applying a function to rows (margin=1)
y<-apply(x,2,prod) # applying a function to columns (margin=2)
y

x<-1:5
y<-1:5
outer(x,y) # returns a matrix with entries obtained by applying a function (the default is the preduct) to all the combinations of the elements of x and y
outer(x,y,"+") # returns a matrix with entries obtained by applying the function sum to all the combinations of the elements of x and y


### Lists ###

Lst <- list(name="Fred", wife="Mary", no.children=3,child.ages=c(4,7,9))
Lst
Lst[[1]]
Lst[[4]]
Lst$child.ages
Lst$child.ages[2]


### Data frame ###

data(airquality) # data set, available in R, with daily air quality measurements in New York from May to September 1973
dim(airquality) # data frame with 154 observations on 6 variables
help(airquality)  #  description of the data set
names(airquality)  # gives all the variable names of the data frame 
head(airquality)  # returns the first 6 rows of the data frame, alternatively use airquality[1:6,]
str(airquality) # summarized the internal structure of an R object

airquality$Ozone
airquality$Ozone[1:5]

airquality$Ozone + airquality$Wind/airquality$Temp
with(airquality, Ozone + Wind/Temp)  # gives the same result as the above command

subset(airquality, airquality$Month == 5 & airquality$Solar.R >= 150)

x<-1:5
y<-factor(c("a","b","a","a","b"))
z<-matrix(rep(7,15),nrow=5,byrow=F)
es.df<-data.frame(z,uno=x,due=y)
es.df

z<-matrix(rep(7,15),nrow=5,byrow=F) # a matrix can be transformed into a data frame
z
class(z)
z_df<- as.data.frame(z)
z_df
class(z_df)

### Loading and saving data sets ###

# dat <- read.table("file.txt", sep=" ", header=T) # to read data from a text file with sep=" " (the default for read.table) as field separator
# dat <- read.table("file.csv", sep=",", header=T) # to read data from a text file with sep="," (the default for read.table) as field separator

airq<-airquality[!is.na(airquality$Ozone),]

write.table(airq,file="airq.txt",sep=" ",row.names=F)  # to save data with white space as separator, row.names=F assures that the row names are not written along airq
write.table(airq,file="airq.csv",sep=",",row.names=F)  # to save with comma as separator

airq1<-read.table("airq.txt",header=T,sep=" ") # in order to load the new data frame
airq2<-read.table("airq.csv",header=T,sep=",") # in order to load the new data frame

attach(airquality)
Ozone[1:3]
detach(airquality)
Ozone[1:3]

### Other useful functions ###

# is.numeric()
# is.vector()
# is.factor()
# is.matrix()
# is.data.frame()
# is.character()

# as.numeric() turns vectors, and matrices of other modes into a numeric ones
# as.character() turns vectors, and matrices of other modes into character ones
# as.integer() turns vectors, and matrices of other modes into integer ones4
# as.factor() will turn a vector, or matrices into factors
# as.matrix() will turn a vector, or data frame into a matrix5
# as.vector() will turn matrices into vectors
# as.data.frame() will turn vectors and matrices into data frames
# as.list() will turn vectors and matrices into lists

### Plotting commands ###

# plot(x,y) if x and y are vectors, plot(x,y) produces a scatterplot of y against x;
# plot(x)  if x is a time series, this produces a time-series plot, if x is a numeric vector, it produces a plot of the values in the vector against their index in the vector.

curve(sin(cos(x)*exp(-x/2)),from=-8,to=7)
curve(x^3-3*x,-2,2)  # high-level plotting command
curve(x^2-2,add=T,col="red")  # low-level plotting command, since with the option add=T the curve is added to an already existing plot

### Functions ###

bmi<-function(weight,height) # Body Mass Index, weight in kilograms and height in metres
  { 
	x<- weight/height^2 
	return(x)
  }

bmi(75,1.7)

x<-c(60,65,75,80,90)
y<-c(1.5,1.6,1.7,1.8,1.9)
bmi(x,y)
bmi(70,y)

bmi1<-function(weight,height,cm=F) # Body Mass Index (height could be in centimetres)
  {
	if(cm==T) height<-height/100
	weight/height ^2
  }

bmi1(75,1.7)
bmi1(75,170,cm=T)

x<-seq(50,100,by=0.1)
y<-seq(1.5,1.95, by=0.01)
index<-outer(x,y,bmi)
persp(x,y,index,xlab="weight",ylab="height",zlab="bmi",ticktype ="detailed") # 3d graphical representation
persp(x,y,index,xlab="weight",ylab="height",zlab="bmi",theta=-90,phi=30,ticktype ="detailed") # rotation

contour(x,y,index,xlab="weight",ylab="height") # contour plot
contour(x,y,index,xlab="weight",ylab="height",col=2:11,levels=20:30) # contour plot with levels from 20 to 30 having different colours
# to add a grid
abline(h=seq(1.5,1.9,by=0.05),lty=2,col="grey")  # low level plot function
abline(v=seq(50,100,by=5),lty=2,col="grey")

geom<-function(n,r,a0)  # the first n terms of a geometric series with initial value a0 and ratio r
  {
	ser<-numeric(n)
	ser[1]<-a0
	for (i in 2:n)
	ser[i]<-ser[i-1]*r
	return(ser)
  }

geom(10,0.5,1)

