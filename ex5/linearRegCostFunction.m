function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h_theta = X * theta;
err = h_theta - y;

sqr_err = err .* err;
J_noReg = 1/(2*m) * sum(sqr_err);

theta_sqr = theta(2:end) .* theta(2:end);
J_reg = lambda/(2*m) * sum(theta_sqr);

J = J_noReg + J_reg;


theta_new = theta;
theta_new(1) = 0;
grad_noReg = 1/m * (err' * X);
grad_reg = lambda/m * theta_new';

grad = grad_noReg + grad_reg;




% =========================================================================

grad = grad(:);

end
