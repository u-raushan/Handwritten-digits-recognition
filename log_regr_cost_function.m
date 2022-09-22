function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION computes cost and gradient for logistic regression with 
%regularization.

% To initialize number of training examples
m = length(y);

% Sigmoid function
h=sigmoid(X*theta);

% Unregularized logistic regression cost
J =-(1/m)*((y'*log(h))+(1-y)'*log(1-h));

% Regularization part
regr=(lambda/(2*m))*sum(theta(2:size(X,2)).^2);

% Regularizes logistic regression cost
J = J+regr;

temp=(h-y)';

% Gradient of the first theta value
grad=(1/m)*(temp*X)';

% Gradient of the rest theta values
grad(2:size(X,2))=grad(2:size(X,2))+(lambda/m)*theta(2:size(X,2));

grad = grad(:);

end
