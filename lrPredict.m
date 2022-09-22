function p = lrPredict(all_theta, X)
%PREDICT predicts the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1).

m = size(X, 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

z=X*all_theta';       
h=sigmoid(z);
[~,p]=max(h,[],2);

end
