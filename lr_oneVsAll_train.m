function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i.

% Size of the matrix X
m = size(X, 1);
n = size(X, 2);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% Set initial theta values to zero
initial_theta=zeros(n+1,1);

options=optimset('GradObj','on','MaxIter',50);

for c=1:num_labels
    [theta]=fmincg(@(t)(lrCostFunction(t, X, (y == c), lambda)),initial_theta,options);
    all_theta(c,:)=theta;
end

end
