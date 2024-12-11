%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CE-M from Wikipedia
% https://en.wikipedia.org/wiki/Cross-entropy_method
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc;

% initial parameters of the distribution (Gaussian)
mu = -2;
sigma2 = 75;

% optimization parameters
max_iters = 100;  % max iterations
N = 50;          % total number of samples
N_elite = 10;     % number of elite samples
epsilon = 0.001;   % stopping criteria

% do the optimization
MU = [mu];
SIGMA = [sigma2];
iter = 0;
tic;
while (iter < max_iters) && (sigma2 > epsilon)

    % Obtain N sample from the current smapling distribution
    X = normrnd(mu, sigma2, N, 1);

    % Evaluate the objective function at the sampled points
    F = zeros(N, 1);
    for i = 1:N
        F(i) = obj_func(X(i));
    end

    % Sort the samples based on the objective function
    [~, idx] = sort(F, 'descend');  % biggest to smallest

    % Update the distribution parameters from elite samples
    F_elite = F(idx(1:N_elite));
    mu = mean(F_elite);
    sigma2 = std(F_elite);

    % store the results
    MU = [MU, mu];
    SIGMA = [SIGMA, sigma2];

    % update iteration count
    iter = iter + 1;
end

% display the results
tot_time = toc;
fprintf('Optimization finished in %d iterations and %.2f seconds\n', iter, tot_time);

figure;
hold on; grid on;
xmin = -6; 
xmax = 6;
ymin = -1;
ymax = 9;
xlim([xmin, xmax]);
ylim([ymin, ymax]);

% plot the objective function
domain = linspace(xmin,xmax, 1000);
S = zeros(1, length(domain));
for i =1:length(domain)
    S(i) = obj_func(domain(i));
end
plot(domain, S, 'b', 'LineWidth', 2);
xlabel('x');
ylabel('S(x)');

% plot the normal distribution
for i = 1:length(MU)
    Y = eval_normal(domain, MU(i), SIGMA(i));
    nrm = plot(domain, Y, 'r', 'LineWidth', 2);
    mean = xline(MU(i), 'g', 'LineWidth', 2);
    pause(1.0);

    msg = sprintf('Iteration: %d, mu: %.2f, sigma2: %.2f', i, MU(i), SIGMA(i));
    title(msg);

    % remove the previous plot
    if i < length(MU)
        delete(nrm);
        delete(mean);
    end
end


% define some arbitrary objectoive function
function S = obj_func(x)
    p = 8.2;
    S = exp(-(x-2)^2) + p * exp(-(x+2)^2);
end

% function to evaluat the normal distribution
function y = eval_normal(X, mu, sigma2)
    len = length(X);
    y = zeros(len, 1);
    for i = 1:len
        y(i) = (1/sqrt(2*pi*sigma2)) * exp(-0.5 * (X(i) - mu)^2 / sigma2);
    end
end
