library(forecast)
library(tseries)
library(zoo)
dt<-read.csv('D:/desktop/Rstl论文/修改意见/夏威夷月度取对数.csv')
ts<-ts(dt[,2],start=c(2005,1),end=c(2023,1),frequency=12)
ts


ndiffs(ts)#判断需要几阶差分才能转化为平稳序列。可见，Nile需要一阶差分后变为平稳序列
dNile <- diff(ts)    #一阶差分                                          
plot(dNile)            #对一阶差分作图
ADF<-adf.test(dNile)
res <- suppressWarnings(adf.test(dNile))
res                    #拒绝原假设：平稳.原假设：存在单位根（存在单位根就是非平稳时间序列）
#2.模型定阶及拟合
fit <- auto.arima(ts)
fit
accuracy(fit)  
#3.模型诊断
qqnorm(fit$residuals)  #画图   
qqline(fit$residuals)  #加线
Box.test(fit$residuals, type="Ljung-Box") #残差检验，不显著：残差平稳
#4.用ARIMA模型做预测
forecast(fit, 1)#预测三期
forecast$Point Forecast
plot(forecast(fit, 3), xlab="Year", ylab="Annual Flow")



