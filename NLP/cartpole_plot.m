%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot DDP Pendulum Results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; clc; close all;

% import helper functions
addpath('../CartPole/');

% load the results
times = readmatrix('results/cartpole_time.csv');
x_t = readmatrix('results/cartpole_states.csv');
u_t = readmatrix('results/cartpole_inputs.csv');

% compute number of time steps and dt
nx = size(x_t,2);
nu = size(u_t,2);
N = size(x_t, 1) - 1;
dt = times(2) - times(1);

% define some dummy dynamics parameters
dyn_params.l = 0.5;   % pendulum length (m)
dyn_params.m = 1.0;   % pendulum mass (kg)

x_lims = [min(x_t(:,2)) - 0.25, max(x_t(:,2)) + 0.25];
y_lims = [-dyn_params.l-0.1, dyn_params.l + 0.2];

figure(1); clf;
hold on;                % <-- important (don’t use hold off)
xline(0); yline(0);
xlabel('x [m]'); ylabel('y [m]');
xlim(x_lims); ylim(y_lims);
axis equal;

realtime_rate = 1.0;    % like your old code (1 = real time)
t0 = tic;

k = 1;
while true
    % target wall time for this frame (use your recorded times if you want)
    % target = times(k) / realtime_rate;        % best if times might be nonuniform
    target = (k-1)*dt / realtime_rate;          % ok if dt is truly constant

    [pole_base, pole, ball, cart] = draw_cartpole(x_t(k,:), dyn_params);
    title(sprintf('Inverted Pendulum at Time t = %.2f s', (k-1)*dt));

    % wait until target time (CPU-friendly)
    pause(max(0, target - toc(t0)));

    delete(pole_base); delete(pole); delete(ball); delete(cart);

    k = k + 1;
    if k == N+1
        k = 1;
        t0 = tic;       % restart timing each cycle
    end
end
