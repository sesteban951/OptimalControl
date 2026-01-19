%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Gradient Estimation via Sampling (1D Example)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; clc; close all;

% come up with a super nonlinear function  
x_vals = linspace(-3,3,500);
y_vals = arrayfun(@(x) f(x), x_vals);
dy_vals = arrayfun(@(x) true_gradient(x), x_vals);

% smoothing scale
params.K = 500;      % number of samples per point
params.sigma = 0.1; % noise standard deviation

% estimate gradient at all eval points (gradient of Gaussian-smoothed f)
grad_est = estimate_gradient(x_vals, params);

% plot the function
figure;

subplot(2,1,1);
hold on; grid on;
xline(0); yline(0);
plot(x_vals, y_vals, 'LineWidth', 2);
title('Function to Estimate Gradient');
xlabel('x'); ylabel('f(x)');

subplot(2,1,2);
hold on; grid on;
xline(0); yline(0);
plot(x_vals, dy_vals, 'b', 'LineWidth', 2, 'DisplayName', 'True Gradient');
plot(x_vals, grad_est, 'r', 'LineWidth', 2, 'DisplayName', 'Estimated Gradient');
title('Gradient Estimation via Sampling');
xlabel('x'); ylabel('df/dx');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Helper FUnctions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% some nasty nonlinear function that is not differentiable everywhere
function y = f(x)

    % piecewise function
    if x < -1
        y = 2 * abs(x + 2) +1;
    elseif x >= -1 && x <= 1
        y = 2*x.^2;
    else
        y = sin(4*x) + 2;
    end
end

% piecewise true gradient of f
function dy = true_gradient(x)

    if x < -1
        % d/dx abs(x+3) = sign(x+3), except undefined at x = -3
        if x < -2
            dy = -2;
        elseif x > -2
            dy = 2;
        else
            dy = 0;  % convention at the kink (could also be NaN)
        end

    elseif x <= 1
        dy = 4*x;   % d/dx

    else
        dy = 4 * cos(4 *x); % d/dx (4 sin(x) + 2)
    end
end

% estimate the gradient via sampling
function dy = estimate_gradient(x, params)

    % preallocate
    dy = zeros(size(x));

    % unpack the params
    K = params.K;
    sigma = params.sigma;

    % loop over all input points
    for i = 1:numel(x)
        
        % choose a sample point
        xi = x(i);

        % Gaussian perturbations
        eps = randn(K,1); % sample a vector of K standard normal variables

        % Antithetic sampling for variance reduction
        eps = [eps; -eps];

        % Sample the function at perturbed points
        fx = arrayfun(@(e) f(xi + sigma*e), eps);

        % Score-function / ES gradient of smoothed objective
        dy(i) = mean(fx .* eps) / sigma;
    end

end