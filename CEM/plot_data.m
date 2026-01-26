%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plot data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc;

% load data
time_dir = "./results/time.csv";
x_opt_dir = "./results/state.csv";
u_opt_dir = "./results/input.csv";
time = readmatrix(time_dir);
x_opt = readmatrix(x_opt_dir);
u_opt = readmatrix(u_opt_dir);
dt = time(2) - time(1);

% extract states
px = x_opt(:,1);
py = x_opt(:,2);
vx = x_opt(:,3);
vy = x_opt(:,4);

plot_mode = 1; % 1: animation, 2: full state and input trajectories

% animation
if plot_mode == 1

    figure;
    px_max = max(px) + 0.5;
    px_min = min(px) - 0.5;
    py_max = max(py) + 0.5;
    py_min = min(py) - 0.5;
    xlim([px_min, px_max]);
    ylim([py_min, py_max]);
    hold on; grid on;
    xline(0);
    yline(0);

    k = 1;
    while true

        % plot the position
        pos = plot(px(k), py(k), 'bo', 'MarkerSize', 8, 'MarkerFaceColor', 'b');
        
        msg = sprintf('Time: %.2f s', time(k));
        title(msg, 'FontSize', 14);

        pause(dt);

        % update index
        if k == length(time)
            k = 1;
        else
            k = k + 1;
        end

        delete(pos);

    end

end

% print full state and input trajectories
if plot_mode == 2

    figure;
    subplot(2,3,1);
    plot(time, px, 'LineWidth', 2);
    xlabel('Time (s)', 'FontSize', 14);
    ylabel('Position x (m)', 'FontSize', 14);

    subplot(2,3,2);
    plot(time, py, 'LineWidth', 2);
    xlabel('Time (s)', 'FontSize', 14);
    ylabel('Position y (m)', 'FontSize', 14);

    subplot(2,3,4);
    plot(time, vx, 'LineWidth', 2);
    xlabel('Time (s)', 'FontSize', 14);
    ylabel('Velocity x (m/s)', 'FontSize', 14);

    subplot(2,3,5);
    plot(time, vy, 'LineWidth', 2);
    xlabel('Time (s)', 'FontSize', 14);
    ylabel('Velocity y (m/s)', 'FontSize', 14);

    subplot(2,3,3);
    plot(time(1:end-1), u_opt(:,1), 'LineWidth', 2);
    xlabel('Time (s)', 'FontSize', 14);
    ylabel('Input u1 (N)', 'FontSize', 14);

    subplot(2,3,6);
    plot(time(1:end-1), u_opt(:,2), 'LineWidth', 2);
    xlabel('Time (s)', 'FontSize', 14);
    ylabel('Input u2 (N)', 'FontSize', 14);

end