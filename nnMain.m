load('data.mat');
m = size(X, 1);

% To load saved matrices from file
load('weights.mat'); 
% Theta1 has size 25 x 401
% Theta2 has size 10 x 26
  
 % To run a pretrained NN model
pred = predict(Theta1, Theta2, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
