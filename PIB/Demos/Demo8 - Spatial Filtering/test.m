function test()

I = imread('lena.gif');

thresh = multithresh(I,19);

valuesMax = [thresh max(I(:))];
quant8_I = imquantize(I, thresh, valuesMax); 

imshowpair(I,quant8_I,'montage');
unique(quant8_I)
end

