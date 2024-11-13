% Model Predictive Path Integral Control (MPPI) for unicycle model
clear all; close all; clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MPPI parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Unicycle
mppi.input_size = 2; % input size
mppi.state_size = 3; % state size

% Rollouts
mppi.K = 250;        % number of rollouts

% Horizon Length
mppi.dt = 0.04;      % time step
mppi.N = 30;         % horizon

% Cost
mppi.lambda = .05;            % tuning weight factor
mppi.mu = [0.0, 0.0];        % input mean     (Gaussian noise)
mppi.sigma = [.2, 1.0];      % input variance (Gaussian noise)
mppi.Q = diag([10,10,5]);    % state cost (for path integral)
mppi.Qf = diag([30,30,20]);  % final state cost (for path integral)
mppi.R = diag([1,.1]);       % state cost (for path integral)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% inital condition
x0 = [0;  % y
      0;  % z
      0]; % theta

% desired state
x_des = [5;     % y
         3;      % z
         pi/2];  % theta

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% simulate
t0 = 0;
tf = 4.0;
[t, x, u] = mppi_control(t0, tf, x0, x_des, mppi);

% animate the solution to the unicycle
fig1 = figure(1);
xlim = [min(x(:, 1)) - 0.05, max(x(:, 1)) + 0.05];
ylim = [min(x(:, 2)) - 0.05, max(x(:, 2)) + 0.05];
pause(1.0);
tic;
for i = 1:length(t)

    while toc < t(i)
        % wait
    end

    % draw the unicycle
    draw_unicycle(t(i) , x(i, :), fig1, xlim, ylim);
end
 
% plot the state data and trajectory
figure(2);
subplot(1,2,1); 
plot(t, x(1:end, :));
yline(x_des(1), 'b--', 'LineWidth', 2);
yline(x_des(2), 'r--', 'LineWidth', 2);
yline(x_des(3), 'y--', 'LineWidth', 2);
ylabel('State');
xlabel('Time (s)');
legend('y', 'z', 'theta');
grid on;

subplot(1,2,2); 
hold on;
plot(t, u(:, 1));
plot(t, u(:, 2));
ylabel('Inputs');
xlabel('Time (s)');
legend('vel', 'omega');
grid on;

figure(3);
hold on;
plot(x(:, 1), x(:, 2), 'LineWidth', 2);
plot(x0(1), x0(2), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
plot(x(end, 1), x(end, 2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
plot(x_des(1), x_des(2), 'go', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('y');
ylabel('z');
axis equal; grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% forward propagate the dynamics with MPPI control
function [t, x, u] = mppi_control(t0, tf, x0, x_des, mppi)

    % simulation time
    tspan = t0: mppi.dt : tf;

    % forward simulate the unicycle with MPPI control
    x_curr = x0;
    x_list = zeros(length(tspan), mppi.state_size);
    u_list = zeros(length(tspan), mppi.input_size);

    % generate the initial nominal control input
    u_nom = zeros(mppi.N, mppi.input_size);
    
    % forward simulate for the entire time scale
    for i = 1:length(tspan)
        
        % print the current sim time
        time_str = sprintf('Sim Time: %.2f', tspan(i));
        disp(time_str);

        % compute the MPPI control input and take the first input
        u_mppi = monte_carlo(x_curr, x_des, u_nom, mppi);
        u = u_mppi(1,:);  % TODO: for this kinemaitc system, consider adding
                          %       some kind of filter so to net get all the jumpiness coming from the noise. 
        
        % save the control input
        u_list(i, :) = u;
        x_list(i, :) = x_curr;
        
        % forward simulate the unicycle for dt
        [~, x_sim] = ode45(@(t, x) unicycle_dynamics_dt(t, x_curr, u), [0:mppi.dt:mppi.dt], x_curr);

        % update the state
        x_curr = x_sim(end, :);
        
        % update the nominal control input
        u_nom = [u_mppi(2:end, :); u_mppi(end, :)];
    end

    % return the solution to the forward prop of the dynamics
    t = tspan;
    x = x_list;
    u = u_list;
end

% perform the K monte carlo rollouts and choose the "averaged" input
function u_mppi = monte_carlo(x0, x_des, u_nom, mppi)

    % initialize the best cost and input containers
    J_list = zeros(mppi.K, 1);
    u_list = zeros(mppi.N, mppi.input_size, mppi.K);

    % perform the monte carlo rollouts
    for k = 1 : mppi.K
        
        % sample a control input
        du_k = sample_control_input(mppi);
        u_k = u_nom + du_k;

        % forward simulate the dynamics
        [t, x] = forward_sim(x0, u_k, mppi);

        % compute the path integral
        J_k = compute_path_integral(x, x_des, u_k, mppi);

        % save the cost and input
        J_list(k) = J_k;
        u_list(:, :, k) = u_k;
    end

    % compute weights
    J_min = min(J_list);
    w_list = exp(-(1/mppi.lambda) * (J_list - J_min));
    % [minValue, minIndex] = min(w_list);

    % compute the numerator from the sum
    num_sum = zeros(mppi.N, mppi.input_size);
    for k = 1:mppi.K
        num_sum = num_sum + w_list(k) * u_list(:, :, k); % NOTE: u_list already has nominal control input in it
    end
    
    u_mppi = num_sum / sum(w_list); % NOTE: nominal control input is already in here, 
                                    % no need to add it again, things will blow up
end

% compute the path integral
function J = compute_path_integral(x_traj, x_des, u_traj, mppi)

    % compute the stage cost
    cost = 0;
    for i = 1 : mppi.N
        
        % compute the nominal cost
        state_cost = (x_traj(i, :)' - x_des)' * mppi.Q * (x_traj(i, :)' - x_des);
        input_cost = u_traj(i, :) * mppi.R * u_traj(i, :)';

        % add the cost to the total cost
        cost = cost + state_cost + input_cost;
    end

    % compute the terminal cost
    terminal_cost = (x_traj(mppi.N+1, :)' - x_des)' * mppi.Qf * (x_traj(mppi.N+1, :)' - x_des);
    cost = cost + terminal_cost;

    % return the total cost
    J = cost;
end

% sample a control input
function du = sample_control_input(mppi)

    % sample a sequence of control inputs
    du_list = zeros(mppi.N, mppi.input_size);
    for i = 1:mppi.N
        
        % sample a random control input TODO: allocate container ahead of tiem and use multivariate 
        u_rand = [];
        for j = 1:mppi.input_size
            u_rand = [u_rand, normrnd(mppi.mu(j), mppi.sigma(j))];
        end

        % save the random control input
        du_list(i,:) = u_rand;
    end

    % return array of control inputs
    du = du_list;    % size (N x input_size)
end

% single rollout given single control tape
function [t, x] = forward_sim(x0, du_list, mppi)

    % forward simulate the dynamics given the control input sequence
    x_curr = x0;
    x_list = zeros(mppi.N+1, mppi.state_size);
    x_list(1, :) = x_curr;

    for i = 1:mppi.N
        % update the zero order hold control input
        u = du_list(i,:);

        % forward simulate the unicycle for dt
        [~, x_sim] = ode45(@(t, x) unicycle_dynamics_dt(t, x_curr, u), [0 : mppi.dt : mppi.dt], x_curr);

        % update the state
        x_curr = x_sim(end, :);

        % save the state after forward prop for dt
        x_list(i+1,:) = x_curr;
    end

    % return the solution to the forward prop of the dynamics
    t = 0 : mppi.dt : mppi.dt*mppi.N;   % size (N+1 x 1)
    x = x_list;      % size (N+1 x state_size)
end

% unicycle dynamics
function xdot = unicycle_dynamics_dt(t, x, u)
    
    % unpack the state
    theta = x(3);

    % input
    v = u(1);
    w = u(2);

    % define the dynamics
    g_x = [cos(theta), 0;
           sin(theta), 0;
           0,          1];
    xdot = g_x * [v; w];
end
