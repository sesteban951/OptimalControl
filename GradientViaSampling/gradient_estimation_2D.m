%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Gradient Estimation via Sampling (2D Example)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; clc; close all;

% Make a grid of evaluation points in R^2
x1 = linspace(-5,5,100);
x2 = linspace(-5,5,100);
[X1, X2] = meshgrid(x1, x2);

% Evaluate f on the grid (for visualization)
y_vals = arrayfun(@(a,b) f([a;b]), X1, X2);
dy_vals = arrayfun(@(x) true_gradient(x), y_vals);

% smoothing scale
params.K = 100;      % number of samples per point
params.sigma = 0.1;  % noise standard deviation

% Estimate gradient at all grid points
G1 = zeros(size(X1));
G2 = zeros(size(X2));
for i = 1:numel(X1)
    x = [X1(i); X2(i)];       
    g = estimate_gradient(x, params);
    G1(i) = g(1);
    G2(i) = g(2);
end
% --- Two-panel figure: (1) surface, (2) 2D vector field ---
figure;

% (1) Left: original surface
subplot(1,2,1); hold on; grid on;
surf(X1, X2, y_vals, 'EdgeColor','none', 'FaceAlpha',0.9);
colorbar;
title('Original surface f(x_1,x_2)');
xlabel('x_1'); ylabel('x_2'); zlabel('f(x_1,x_2)');
view(45,30);

% (2) Right: 2D vector field on contour map (sampling gradient)
subplot(1,2,2); hold on; grid on;
contourf(X1, X2, y_vals, 30); colorbar;

% downsample arrows so it isn’t cluttered
step = 4;
Xs  = X1(1:step:end, 1:step:end);
Ys  = X2(1:step:end, 1:step:end);
G1s = G1(1:step:end, 1:step:end);
G2s = G2(1:step:end, 1:step:end);

% (optional) normalize arrow lengths for readability
mag = sqrt(G1s.^2 + G2s.^2);
mag(mag == 0) = 1;
U = G1s ./ mag;
V = G2s ./ mag;

quiver(Xs, Ys, U, V, 0.8, 'k');
title('2D vector field: sampling-based gradient direction');
xlabel('x_1'); ylabel('x_2');
axis tight;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Helper FUnctions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% some nasty nonlinear function that is not differentiable everywhere
function y = f(x)

    % unpack
    x1 = x(1); 
    x2 = x(2);

    % nonsmooth term + oscillatory term + piecewise bump
    if x1 + x2 < 0
        y = abs(x1 - 1) + 0.5*abs(x2 + 2);
    else
        y = 2*sin(0.8*x1) + 1.5*cos(0.6*x2);
    end

    % add a quadratic well for structure
    y = y + 0.05*(x1^2 + x2^2);

end

% piecewise true gradient of f
function dy = true_gradient(x)
    if x < -1
        % f(x) = abs(x+3)
        if x < -3
            dy = -1;
        elseif x > -3
            dy = 1;
        else
            dy = NaN; % or 0 if you prefer a convention
        end
    elseif x <= 1
        % f(x) = 3x^2
        dy = 6*x;
    else
        % f(x) = 4 sin(x) + 2
        dy = 4*cos(x);
    end
end

% estimate the gradient via sampling (returns 2x1 vector)
function dy = estimate_gradient(x, params)
    x = x(:);
    assert(numel(x)==2, 'estimate_gradient expects a 2D point x = [x1; x2].');

    K = params.K;
    sigma = params.sigma;

    eps = randn(K, 2);
    eps = [eps; -eps];  % (2K)x2

    fx = zeros(size(eps,1), 1);
    for k = 1:size(eps,1)
        xk = x + sigma * eps(k,:)';
        fx(k) = f(xk);
    end

    dy = (eps' * fx) / (size(eps,1) * sigma); % 2x1
end