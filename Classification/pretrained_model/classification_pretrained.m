% % Resize images in data store using custom reader function
clc, close all, clear all

filePaths = cell(1360, 1); % Preallocate cell array
folder = '.. /17flowers';
fileList = dir(fullfile(folder, '*.jpg'));
for i = 1:numel(fileList)
    filePaths{i} = fullfile(folder, fileList(i).name);
end

labels = repmat(1:17, 80, 1);
labels = labels(:);

imds = imageDatastore(filePaths, 'Labels', categorical(labels));
imds.ReadFcn = @customreader;
reset(imds);

tbl = countEachLabel(imds)

% Split the dataset into training, validation, and testing sets
[trainImgs, ValImgs, testImgs] = splitEachLabel(imds, 0.7, 0.15, 0.15, 'randomized');

% Load the pre-trained network
net = alexnet;

% Replace the last layer to match the number of classes in the data
numClasses = numel(categories(imds.Labels));
layers = net.Layers;
layers(end-2) = fullyConnectedLayer(numClasses);
layers(end) = classificationLayer;

% Set the training options
options = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 1e-4, ...
    'MiniBatchSize', 16, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', ValImgs, ...
    'ValidationFrequency', 3, ...
    'Verbose', false, ...
    'Plots', 'training-progress');


% Train the CNN using the training set and validate using the validation set
net = trainNetwork(trainImgs, layers, options);

deepNetworkDesigner(net)

%save the network in case we want to use it again
save('classification_net_pretrained.mat', 'net') %filename, variable
%can load with 'load mynet.mat'

% Evaluate the performance of the trained CNN on the test set
predictedLabels = classify(net, testImgs);
trueLabels = testImgs.Labels;
accuracy = sum(predictedLabels == testImgs.Labels)/numel(testImgs.Labels) * 100;
disp(['Accuracy for test images: ' num2str(accuracy) '%']);

% Display an example image and its predicted label
idx = randi(numel(testImgs.Files));
I = readimage(testImgs, idx);
imshow(I);
title(['True Label: ' char(trueLabels(idx)) ', Predicted Label: ' char(predictedLabels(idx))]);

% Plot a confusion matrix
figure;
plotconfusion(trueLabels, predictedLabels);

function data = customreader(filename)
onState = warning('off', 'backtrace');
c = onCleanup(@() warning(onState));
data = imread(filename);
data = data(:,:,min(1:3, end)); 
data = imresize(data, [227 227]);
end

