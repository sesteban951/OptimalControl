%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot DP Results for Inverted Pendulum
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; clc; close all;

% load results from DP
load('results.mat', 'V', 'pi_star', ...
                    'x1_grid', 'x2_grid', 'u_grid', ...
                    'x_des', ...
                    'dt', 'N', ...
                    'dyn_params', 'cost_params');

% initial state
x0 = [0; 0]; 

% make time span
tspan = 0:dt:(N*dt);

% plotting options
plot_setting = 2; % 1: pendulum, 2: value function, 3: pi_star
value_plot_type = 'surface'; % 'heatmap' or 'surface' for value function

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PLOT PENDULUM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% plot the pendulum animation
if plot_setting == 1

    % make the dynamics functions
    [f_cont, f_disc] = make_dynamics_functions(dyn_params);

    % import helper functions
    addpath('../InvertedPendulum/');

    % forward simulate under optimal policy
    x_t = zeros(2, N+1);
    u_t = zeros(1, N);
    x_t(:,1) = x0;
    xk = x0;
    for k = 1:N
        
        % wrap/clamp + fast nearest-neighbor indexing (consistent with DP)
        [idx1, idx2] = nearest_grid_index(xk, x1_grid, x2_grid);

        % get optimal control from policy
        uk = pi_star(idx1, idx2, k);
        u_t(k) = uk;

        % simulate one step
        xk = f_disc(xk, uk);
        xk(1) = wrap_to_pi(xk(1)); % keep angle sane
        x_t(:,k+1) = xk;
    end

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
        [base, pole, ball] = draw_pendulum(time, x_t(:,k), dyn_params);

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

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PLOT VALUE FUNCTION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% animate the value function over time
if plot_setting == 2

    figure(2); clf;

    % make mesh for surface plot (note: need omega x theta ordering)
    [TH, OM] = meshgrid(x1_grid, x2_grid);

    % initial slice
    V0 = squeeze(V(:,:,1))';   % size [x2_N x x1_N]

    if strcmpi(value_plot_type, 'surface')
        % --- surface plot ---
        Vk_plot = surf(TH, OM, V0);
        shading interp;
        view(3);
        xlabel('Angle (rad)');
        ylabel('Angular Velocity (rad/s)');
        zlabel('Value');
        colorbar;
        hold on;
        plot3(x_des(1), x_des(2), max(V0(:)), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
        hold off;
    elseif strcmpi(value_plot_type, 'heatmap')
        % --- heatmap ---
        Vk_plot = imagesc(x1_grid, x2_grid, V0);
        set(gca, 'YDir', 'normal'); % optional
        hold on;
        plot(x_des(1), x_des(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
        hold off;
        xlabel('Angle (rad)');
        ylabel('Angular Velocity (rad/s)');
        colorbar;
    end

    % set axis limits to the grid limits
    xlim([min(x1_grid) max(x1_grid)]);
    ylim([min(x2_grid) max(x2_grid)]);

    k = 1;
    while true

        % extract value function at time step k
        Vk = squeeze(V(:,:,k))';  % [x2_N x x1_N]

        % update plot data depending on plot type
        if strcmpi(value_plot_type, 'surface')
            Vk_plot.ZData = Vk;
        else
            Vk_plot.CData = Vk;
        end

        % update title
        time = dt*(k-1);
        msg = sprintf('Value Function at Time t = %.2f s', time);
        title(msg);

        pause(dt);

        % increment or reset k
        k = k + 1;
        if k == N+1
            k = 1;
        end
    end

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PLOT OPTIMAL POLICY
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% animate the optimal policy over time
if plot_setting == 3

    figure(3);
    hold on;

    % initial policy slice
    U0 = squeeze(pi_star(:,:,1))';
    Uk_plot = imagesc(x1_grid, x2_grid, U0);
    xdes_plot = plot(x_des(1), x_des(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2); % desired state
    xlabel('Angle (rad)');
    ylabel('Angular Velocity (rad/s)');
    colorbar;

    % keep the colormap scale fixed to torque bounds
    caxis([min(u_grid) max(u_grid)]);

    % set axis limits to the grid limits
    xlim([min(x1_grid) max(x1_grid)]);
    ylim([min(x2_grid) max(x2_grid)]);

    k = 1;
    while true

        % extract policy at time step k (k = 1..N)
        Uk = squeeze(pi_star(:,:,k))';
        Uk_plot.CData = Uk;

        % update title
        time = dt*(k-1);
        msg = sprintf('Optimal Policy \\pi^* at Time t = %.2f s', time);
        title(msg);

        % refresh plot
        pause(dt);

        % increment or reset k
        k = k + 1;
        if k == N+1
            k = 1;   % pi_star is only defined up to N
        end
    end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Helper Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% dynamics function
function [f_cont, f_disc] = make_dynamics_functions(params)

    % extract the parameters
    m = params.m;
    b = params.b;
    l = params.l;
    g = params.g;
    dt = params.dt;

    % Continuous-time dynamics
    f_cont = @(x,u) [x(2);
                     (1/(m*l^2)) * (-b*x(2) - m*g*l*sin(x(1)) + u)];

    % Discrete-time dynamics via RK4
    f_disc = @(x,u) rk4(f_cont, x, u, dt);
end

% RK4 integrator
function x_next = rk4(f_cont, x, u, dt)

    % compute vectors
    k1 = f_cont(x, u);
    k2 = f_cont(x + 0.5*dt*k1, u);
    k3 = f_cont(x + 0.5*dt*k2, u);
    k4 = f_cont(x + dt*k3, u);

    % next state
    x_next = x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4);
end

% find nearest grid index for a given state (fast for uniform grids)
function [i,j] = nearest_grid_index(x, x1_grid, x2_grid)

    % wrap angle and clamp omega
    th = wrap_to_pi(x(1));
    om = min(max(x(2), x2_grid(1)), x2_grid(end));

    % grid sizes and spacings
    n1 = numel(x1_grid);
    n2 = numel(x2_grid);
    d1 = x1_grid(2) - x1_grid(1);
    d2 = x2_grid(2) - x2_grid(1);

    % nearest-index formula (1-based)
    i = 1 + round((th - x1_grid(1)) / d1);
    j = 1 + round((om - x2_grid(1)) / d2);

    % clamp indices to valid range
    i = min(max(i, 1), n1);
    j = min(max(j, 1), n2);
end

% wrap angle to [-pi, pi)
function angle_wrapped = wrap_to_pi(angle)
    angle_wrapped = mod(angle + pi, 2*pi) - pi;
end