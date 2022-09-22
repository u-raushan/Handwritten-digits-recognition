% To load the data
load('ex3data1.mat');

# To display 100 random data points 
m = size(X, 1);
rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);
displayData(sel);
