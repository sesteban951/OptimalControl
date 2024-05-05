% Model Predictive Path Integral Control (MPPI) for Cartpole model
clear all; close all; clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MPPI parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Cartpole
mppi.state_size = 4; % state size
mppi.input_size = 1; % input size

% Rollouts
mppi.K = 500;        % number of rollouts

% Horizon Length
mppi.dt = 0.01;      % time step
mppi.N = 75;         % horizon

% Cost
mppi.lambda = 0.01;         % tuning weight factor
mppi.mu = [0.0];            % input mean     (Gaussian noise)
mppi.sigma = [40.0];        % input variance (Gaussian noise)
mppi.Q = diag([1,50,50,5]);  % state cost (for path integral)
mppi.Qf = diag([0,0,0,0]); % final state cost (for path integral)
mppi.R = diag([50]);         % state cost (for path integral)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Dyanmics parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% inital condition
x0 = [0;  % z
      0;  % theta
      0;  % zdot
      0]; % thetadot

% desired state
x_des = [0;     % z
         pi;  % theta
         0;     % zdot
         0];    % thetadot

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% simulate
t0 = 0;
tf = 5.0;
[t, x, u] = mppi_control(t0, tf, x0, x_des, mppi);

% animate the cartpole
figure(1);
pause(1.0);
tic;
for i = 1:length(t)

    while toc < t(i)
        % wait
    end

    % draw the unicycle
    draw_cartpole(t(i) , x(i, :))
end

% plot the state and input data
figure(2);
subplot(2,1,1);
plot(t, x(1:end, :), 'LineWidth', 2);
yline(x_des(1), 'b--', 'LineWidth', 2);
yline(x_des(2), 'r--', 'LineWidth', 2);
yline(x_des(3), 'y--', 'LineWidth', 2);
yline(x_des(4), 'g--', 'LineWidth', 2);
ylabel('State');
xlabel('Time (s)');
legend('$z$', '$\theta$', '$\dot{z}$', '$\dot{\theta}$', 'Interpreter', 'latex');
grid on;

subplot(2,1,2);
stairs(t, u(1:end, :), 'LineWidth', 2);
ylabel('Input');
xlabel('Time (s)');
legend('$u$', 'Interpreter', 'latex');
grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% forward propagate the dynamics with MPPI control
function [t, x, u] = mppi_control(t0, tf, x0, x_des, mppi)

    % simulation time
    tspan = t0: mppi.dt : tf;

    % forward simulate the dynamics with MPPI control
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
        u = u_mppi(1,:);
        
        % save the control input
        u_list(i, :) = u;
        x_list(i, :) = x_curr;
        
        % forward simulate the dynamics for dt
        [~, x_sim] = ode45(@(t, x) dynamics_dt(t, x_curr, u), [0:mppi.dt:mppi.dt], x_curr);

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
        num_sum = num_sum + w_list(k) * u_list(:, :, k);
    end
    
    u_mppi = num_sum / sum(w_list);
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

        % forward simulate the dynamics for dt
%         [~, x_sim] = ode45(@(t, x) dynamics_dt(t, x_curr, u), [0 : mppi.dt : mppi.dt], x_curr);
        x_sim = x_curr + dynamics_dt(0, x_curr, u) * mppi.dt;
        x_sim = x_sim';

        % update the state
        x_curr = x_sim(end, :);

        % save the state after forward prop for dt
        x_list(i+1,:) = x_curr;
    end

    % return the solution to the forward prop of the dynamics
    t = 0 : mppi.dt : mppi.dt*mppi.N;   % size (N+1 x 1)
    x = x_list;      % size (N+1 x state_size)
end

% system dynamics
function xdot = dynamics_dt(t, x, u)

    % model parameters
    mc = 1.0;  % cart mass
    mp = 0.3;  % pole mass
    l = 1.0;   % pole length
    g = 9.81;  % gravity
    bc = 0.0;  % cart damping
    bp = 0.0;  % pole damping

    % unpack the state
    th = x(2);
    z_dot = x(3);
    th_dot = x(4);

    % convert to configuration space
    q_dot = [z_dot; th_dot];

    % Matrix info
    D = [(mc+mp),      mp*l*cos(th); 
         mp*l*cos(th), mp*(l^2)];
    
    C = [bc, -mp*l*th_dot*sin(th); 
         0, bp];
    
    G = [0; 
         mp*g*l*sin(th)];

    B = [1; 
         0];
    
    % set dynamics info
    f_x = [q_dot;
            -inv(D)*(C*q_dot + G)];
    g_x = [zeros(2,1);
            inv(D)*B];

    % compute the dynamics
    xdot = f_x + g_x * u;
end
