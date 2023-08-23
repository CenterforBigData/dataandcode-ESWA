library(stR)
library(forecast)
library(ggplot2)

dt<-read.csv('D:/desktop/香港/新加坡取对数.csv')

y<-ts(dt[,2],start=c(2001,1),frequency=12)

y
 

plot(y)
a<-str(y)
a
plot(str(y,t.window=365,s.window='periodic',robust=TRUE))

m <- decompose(y)
plot(m)
dec<- AutoSTR(y)
plot(dec)

STR <- components(dec)
plot(STR)
comp(Trend)
library(xlsx)
write.xlsx(comp, file ='D:/desktop/香港/泰国str.xlsx')
comp$Trend
