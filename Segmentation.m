clc, close all, clear all

classNames = ["flower" "background"];
pixelLabelID = [1 3];
pxds = pixelLabelDatastore('daffodilSeg\train\pxds\',classNames,pixelLabelID);
imds = imageDatastore('daffodilSeg\train\imds\');

pxdsVal = pixelLabelDatastore('daffodilSeg\val\pxds\',classNames,pixelLabelID);
imdsVal = imageDatastore('daffodilSeg\val\imds\');

I = readimage(imds,1);
C = readimage(pxds,1);
C(5,5)
B = labeloverlay(I,C);
%figure
%imshow(B)

inputSize = [256 256 3];
numClasses = numel(classNames);

lgraph = deeplabv3plusLayers(inputSize, numClasses, "resnet18");

dsVal = combine(imdsVal,pxdsVal);

options = trainingOptions('sgdm', ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',10,...
    'LearnRateDropFactor',0.3,...
    'Momentum',0.9, ...
    'InitialLearnRate',1e-3, ...
    'L2Regularization',0.005, ...
    'ValidationData',dsVal,...
    'MaxEpochs',3, ...  
    'MiniBatchSize',2, ...
    'Shuffle','every-epoch', ...
    'VerboseFrequency',1,...
    'Plots','training-progress');


trainingData = combine(imds,pxds);

net = trainNetwork(trainingData,lgraph,options);

testImage = imread('daffodilSeg\test\imds\image_0079.png');
imshow(testImage)

C = semanticseg(testImage,net);
B = labeloverlay(testImage,C);
imshow(B)