close all;
clear;
clc;

imds = imageDatastore('daffodilSeg\ImagesRsz256\');

classNames = ["flower" "background"];

pixelLabelID = [1 3];

%groundtruth labels
pxds = pixelLabelDatastore('daffodilSeg\LabelsRsz256',classNames,pixelLabelID);


% Define the downsampling layers
downsamplingLayers = [
    convolution2dLayer(3, 64, 'Padding', 1) 
    reluLayer() % ReLU activation function
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, 64, 'Padding', 1)
    reluLayer()
    maxPooling2dLayer(2, 'Stride', 2)
];

% Define the upsampling layers
upsamplingLayers = [
    transposedConv2dLayer(4, 64, 'Stride', 2, 'Cropping', 1)
    reluLayer()
    transposedConv2dLayer(4, 64, 'Stride', 2, 'Cropping', 1)
    reluLayer()
];

% Define the output layers
finalLayers = [
    convolution2dLayer(1, 2)
    softmaxLayer()
    pixelClassificationLayer()
];

% Stack all layers together
net = [
    imageInputLayer([256 256 3])
    downsamplingLayers 
    upsamplingLayers
    finalLayers
];

opts = trainingOptions('sgdm', ...
    'InitialLearnRate',1e-4, ...
    'MaxEpochs',100, ... % was 100 - training cycles
    'MiniBatchSize',8, ...
    'Plots', 'training-progress');

trainingData = combine(imds,pxds);

net = trainNetwork(trainingData,net,opts);

deepNetworkDesigner(net)

%save the network in case we want to use it again
save('segmentnet.mat', 'net') %filename, variable
%can load with 'load mynet.mat'


%----------------------Eval-------------------------
imds_test = imageDatastore('daffodilSeg\Test\');
pxds_test = pixelLabelDatastore('daffodilSeg\Test_ground_truth',classNames,pixelLabelID);

%show first test image
T = readimage(imds_test,1);
figure
imshow(T)

%Do segmentation, save output images to disk
pxdsResults = semanticseg(imds_test,net,"WriteLocation","out");

%show a couple of output images, overlaid
overlayOut = labeloverlay(readimage(imds_test,1),readimage(pxdsResults,1)); %overlay
figure
imshow(overlayOut);
title('overlayOut')

overlayOut = labeloverlay(readimage(imds_test,2),readimage(pxdsResults,2)); %overlay
figure
imshow(overlayOut);
title('overlayOut2')

% evaluate segmentation results on a per-image basis
imageMetrics = evaluateSemanticSegmentation(pxdsResults, pxds_test, 'Verbose', true);

% get the IoU scores for each image
iouScores = imageMetrics.ImageMetrics.MeanIoU;

figure
cm = confusionchart(imageMetrics.ConfusionMatrix.Variables, ...
  classNames, Normalization='row-normalized');

cm.Title = 'Normalized Confusion Matrix (%)';

imageIoU = imageMetrics.ImageMetrics.MeanIoU;
figure
histogram(imageIoU)
title('Image Mean IoU')

% sort the images by their IoU scores
[sortedIoU, idx] = sort(iouScores);
bestIdx = idx(end);
worstIdx = idx(1);

disp(sortedIoU)

% display the best and worst images
bestImage = readimage(imds_test, bestIdx);
worstImage = readimage(imds_test, worstIdx);

bestOverlay = labeloverlay(bestImage, readimage(pxdsResults, bestIdx));
worstOverlay = labeloverlay(worstImage, readimage(pxdsResults, worstIdx));

figure;
subplot(2,2,1);
imshow(bestImage);
title('Best Image');

subplot(2,2,2);
imshow(bestOverlay);
title(sprintf('Best Overlay (IoU = %.5f)', sortedIoU(end)));

subplot(2,2,3);
imshow(worstImage);
title('Worst Image');

subplot(2,2,4);
imshow(worstOverlay);
title(sprintf('Worst Overlay (IoU = %.5f)', sortedIoU(1)));