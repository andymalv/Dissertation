function dataOut = IMUFilter(dataIn, n, Wn)

[b,a] = butter(n, Wn);

dataOut = filter(b,a,dataIn);


end