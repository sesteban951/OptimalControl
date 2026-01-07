%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot DDP Pendulum Results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; clc; close all;

% import helper functions
addpath('../InvertedPendulum/');

% load the results
times = readmatrix('results/pendulum_time.csv');
x_t = readmatrix('results/pendulum_states.csv');
u_t = readmatrix('results/pendulum_inputs.csv');

% compute number of time steps and dt
nx = size(x_t,2);
nu = size(u_t,2);
N = size(x_t, 1) - 1;
dt = times(2) - times(1);

% define some dummy dynamics parameters
dyn_params.l = 1.0;   % pendulum length (m)
dyn_params.m = 1.0;   % pendulum mass (kg)

figure(1);
hold off;
xlim([-1.25 1.25]);
ylim([-1.25 1.25]);
axis equal;

k = 1;
while true

    % get current time
    time = dt*(k-1);

    % draw the pendulum at time step k
    [base, pole, ball] = draw_pendulum(time, x_t(k, :), dyn_params);

    % update title
    msg = sprintf('Inverted Pendulum at Time t = %.2f s', time);
    title(msg);

    % pause for dt seconds
    pause(dt);

    % delete previous drawings
    delete(base);
    delete(pole);
    delete(ball);

    % increment or reset k
    k = k + 1;
    if k == N+1
        k = 1;
    end
end