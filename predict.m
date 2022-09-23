function p = predict(Theta1, Theta2, X)
%PREDICT predicts the label of an input given a trained neural network

% Useful values
m = size(X, 1);
X=[ones(m,1) X];

% Hidden layer
layer2=sigmoid(X*Theta1');
layer2=[ones(m,1) layer2];

% Output layer
layer3=sigmoid(layer2*Theta2');
[~,p]=max(layer3,[],2);

end
