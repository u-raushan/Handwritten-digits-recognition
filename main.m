% To load the data
load('ex3data1.mat');

# To display 100 random data points 
m = size(X, 1);
rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);
displayData(sel);

% To train the model for all classes at once 
num_labels = 10; % 10 labels, from 1 to 10 
lambda = 0.1;
[all_theta] = lrTrain(X, y, num_labels, lambda);

% To predict the digit and show probability
pred = lrPredict(all_theta, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
