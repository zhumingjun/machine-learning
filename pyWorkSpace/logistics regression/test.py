from numpy import *
import logRegres
dataMat,labelMat=logRegres.loadDataSet();
weights=logRegres.stocGradAscent0(array(dataMat),labelMat)
print(weights)
logRegres.plotBestFit(weights)