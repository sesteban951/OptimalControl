%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Gradient Estimation via Sampling (2D Example)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; clc; close all;

% Make a grid of evaluation points in R^2
x1 = linspace(-5,5,200);
x2 = linspace(-5,5,200);
[X1, X2] = meshgrid(x1, x2);

% Evaluate f on the grid (for visualization)
y_vals = arrayfun(@(a,b) f([a;b]), X1, X2);

% True gradient on the grid (for reference / comparison)
G1_true = zeros(size(X1));
G2_true = zeros(size(X2));
for i = 1:numel(X1)
    g = true_gradient([X1(i); X2(i)]);
    G1_true(i) = g(1);
    G2_true(i) = g(2);
end

% smoothing scale
params.K = 100;      % number of samples per point
params.sigma = 0.05;  % noise standard deviation

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
step = 10;
Xs  = X1(1:step:end, 1:step:end);
Ys  = X2(1:step:end, 1:step:end);
G1s = G1(1:step:end, 1:step:end);
G2s = G2(1:step:end, 1:step:end);
G1ts = G1_true(1:step:end, 1:step:end);
G2ts = G2_true(1:step:end, 1:step:end);

% normalize sampled-gradient arrow lengths
mag = sqrt(G1s.^2 + G2s.^2);
mag(mag == 0) = 1;
U = G1s ./ mag;
V = G2s ./ mag;

% normalize true-gradient arrow lengths (NaNs propagate -> arrow skipped)
magt = sqrt(G1ts.^2 + G2ts.^2);
magt(magt == 0) = 1;
Ut = G1ts ./ magt;
Vt = G2ts ./ magt;

h_samp = quiver(Xs, Ys, U,  V,  0.8, 'k', 'LineWidth', 1.0);
h_true = quiver(Xs, Ys, Ut, Vt, 0.8, 'r', 'LineWidth', 1.0);
legend([h_samp, h_true], {'sampled gradient', 'true gradient'}, ...
       'Location', 'best');
title('2D vector field: sampled (black) vs. true (red) gradient');
xlabel('x_1'); ylabel('x_2');
axis tight;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Second figure: Gaussian-smoothed surface
%   f_sigma(x) = E_eps[ f(x + sigma * eps) ]
%   This is exactly the function whose gradient the sampling estimator
%   is approximating.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% smoothing parameters (separate from gradient-estimation params)
params_smooth.K     = 50;   % samples per point (fewer than gradient -> faster)
params_smooth.sigma = 0.5;  % smoothing scale (try 0.1 ... 1.0)
sigma_smooth = params_smooth.sigma;

% Estimate the smoothed function on the grid
y_smooth = zeros(size(X1));
for i = 1:numel(X1)
    y_smooth(i) = estimate_function([X1(i); X2(i)], params_smooth);
end

figure;

% (1,1) True surface
subplot(2,2,1); hold on; grid on;
surf(X1, X2, y_vals, 'EdgeColor','none', 'FaceAlpha',0.9);
colorbar;
title('True surface f(x_1,x_2)');
xlabel('x_1'); ylabel('x_2'); zlabel('f(x_1,x_2)');
view(45,30);

% (1,2) True contour
subplot(2,2,2); hold on; grid on;
contourf(X1, X2, y_vals, 30); colorbar;
title('True contour of f');
xlabel('x_1'); ylabel('x_2');
axis tight;

% (2,1) Smoothed surface
subplot(2,2,3); hold on; grid on;
surf(X1, X2, y_smooth, 'EdgeColor','none', 'FaceAlpha',0.9);
colorbar;
title(sprintf('Smoothed surface f_\\sigma,  \\sigma = %.2f', sigma_smooth));
xlabel('x_1'); ylabel('x_2'); zlabel('f_\\sigma(x_1,x_2)');
view(45,30);

% (2,2) Smoothed contour
subplot(2,2,4); hold on; grid on;
contourf(X1, X2, y_smooth, 30); colorbar;
title(sprintf('Smoothed contour of f_\\sigma,  \\sigma = %.2f', sigma_smooth));
xlabel('x_1'); ylabel('x_2');
axis tight;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Helper FUnctions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% piecewise function: sharp fold (|.|), cliff (jump), and a flat plateau
function y = f(x)

    % unpack
    x1 = x(1);
    x2 = x(2);

    % --- base layer ---
    % sharp V-fold along the line x1 + x2 = 0 (non-differentiable kink)
    fold = 1.5 * abs(x1 + x2);
    % gentle quadratic bowl so things slope toward the origin
    bowl = 0.15 * (x1^2 + x2^2);
    y = fold + bowl;

    % --- flat ridge: x1 in [1, 5] is a constant plateau, higher than the rest ---
    %     base layer max (over x1 in [-5, 1]) is ~22.5, so 25 sits above everything
    if x1 >= 1
        y = 25;
    end

    % --- second cliff: circular pit (drop) centered at (-2, 2) ---
    if (x1 + 2)^2 + (x2 - 2)^2 < 1.2^2
        y = y - 3;
    end

    % --- flat plateau: constant value in a rectangular region ---
    %     (zero gradient here -> sampling estimator should return ~0)
    if x1 >= -4 && x1 <= -1.5 && x2 >= -4.5 && x2 <= -2
        y = 2;
    end

end

% Piecewise true gradient of f. Returns 2x1 vector [df/dx1; df/dx2].
% Returns [NaN; NaN] at non-differentiable points (jumps and the |.| seam).
function dy = true_gradient(x)
    x1 = x(1);
    x2 = x(2);

    % --- flat ridge: x1 in [1, 5], y = 25 (zero gradient) ---
    if x1 >= 1
        dy = [0; 0];
        return;
    end

    % --- flat plateau: y = 2 (zero gradient) ---
    if x1 >= -4 && x1 <= -1.5 && x2 >= -4.5 && x2 <= -2
        dy = [0; 0];
        return;
    end

    % --- base layer: y = 1.5*|x1 + x2| + 0.15*(x1^2 + x2^2) ---
    %     (the circular pit only adds a constant offset inside, so the
    %      gradient there equals the base-layer gradient; the boundary
    %      itself is a jump and is non-differentiable)
    s = x1 + x2;
    if s == 0
        dy = [NaN; NaN];   % kink along x1 + x2 = 0
    else
        dy = [1.5*sign(s) + 0.3*x1;
              1.5*sign(s) + 0.3*x2];
    end
end

% estimate the Gaussian-smoothed function value via sampling
%   f_sigma(x) = E_eps[ f(x + sigma * eps) ],  eps ~ N(0, I)
function y = estimate_function(x, params)
    x = x(:);
    assert(numel(x)==2, 'estimate_function expects a 2D point x = [x1; x2].');

    K = params.K;
    sigma = params.sigma;

    eps = randn(K, 2);
    eps = [eps; -eps];  % antithetic samples (2K)x2

    fx = zeros(size(eps,1), 1);
    for k = 1:size(eps,1)
        xk = x + sigma * eps(k,:)';
        fx(k) = f(xk);
    end

    y = mean(fx);
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