function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C_options = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_options = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

errs = zeros(length(C_options) * length(sigma_options),3);
for i_c = 1:length(C_options)
	for i_sigma = 1:length(sigma_options)
		C = C_options(i_c);
		sigma = sigma_options(i_sigma);
		
		model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
		predictions = svmPredict(model, Xval);
		err = mean(double(predictions ~= yval));
		idx = (i_c - 1) * length(C_options) + i_sigma;
		errs(idx, 1) = C;
		errs(idx, 2) = sigma;
		errs(idx, 3) = err;
		fprintf(['%f, C = %f, sigma = %f, err = %f' ], idx, C, sigma, err);
	end
end
[minval, idx] = min(errs(:,3));
C = errs(idx,1);
sigma = errs(idx,2);


% =========================================================================

end
