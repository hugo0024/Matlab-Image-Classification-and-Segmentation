% % Resize images in data store using custom reader function
clc, close all, clear all

% % Read datastore already provided in MATLAB 
imds = imageDatastore('17flowers','IncludeSubfolders',true,'LabelSource',...
    'foldernames');
imds.ReadFcn = @customreader;
reset(imds);

tbl = countEachLabel(imds)

% Split the dataset into training, validation, and testing sets
[trainImgs, valImgs, testImgs] = splitEachLabel(imds, 0.7, 0.2, 0.1, 'randomized');

% Define the CNN architecture
layers = [
    imageInputLayer([224 224 3])
    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    fullyConnectedLayer(128)
    reluLayer
    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(32)
    reluLayer
    fullyConnectedLayer(17)
    softmaxLayer
    classificationLayer];

% Set the training options
options = trainingOptions('adam', ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 0.001, ...
    'ValidationData', valImgs, ...
    'ValidationFrequency', 10, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Train the CNN using the training set and validate using the validation set
net = trainNetwork(trainImgs, layers, options);

% Evaluate the performance of the trained CNN on the test set
predictedLabels = classify(net, testImgs);
accuracy = sum(predictedLabels == testImgs.Labels)/numel(testImgs.Labels);
fprintf('Accuracy: %.2f%%\n', accuracy*100);

deepNetworkDesigner(net)

function data = customreader(filename)
onState = warning('off', 'backtrace');
c = onCleanup(@() warning(onState));
data = imread(filename);
data = data(:,:,min(1:3, end)); 
data = imresize(data, [224 224]);
end

