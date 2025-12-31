%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Dynamic Programming (DP) for Pendulum Swing-Up
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; clc; close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SETUP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%desired states
x_des = [pi/2; 0];

% simulation parameters
tf = 3;
dt = 0.05;
tspan = 0:dt:tf;
N = numel(tspan) - 1;

% state discretization
x1_min = -pi; % theta min
x1_max =  pi; % theta max
x2_min = -10;  % theta_dot min
x2_max =  10;  % theta_dot max
x1_N = 101;    % number of grid points for theta
x2_N = 101;    % number of grid points for theta_dot
x1_grid = linspace(x1_min, x1_max, x1_N);
x2_grid = linspace(x2_min, x2_max, x2_N);

% input discretization
u_min = -10; % min torque
u_max = 10;  % max torque
u_N = 101;    % number of grid points for torque
u_grid = linspace(u_min, u_max, u_N);

% create the dynamics function
dyn_params.m = 1;
dyn_params.b = 0.01;
dyn_params.l = 1;
dyn_params.g = 9.81;
dyn_params.dt = dt;
[f_cont, f_disc] = make_dynamics_functions(dyn_params);

% create the cost function 
cost_params.Q = diag([20, 1]);    % state cost weights
cost_params.R = 0.01;             % input cost weight
cost_params.Qf = diag([200, 10]); % final state cost weights
[c_stage, c_terminal] = make_cost_functions(cost_params, x_des);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% VALUE FUNCTION INITIALIZATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% build the value function 
% V(i,j,k) = value at grid state (x1_grid(i), x2_grid(j)) at time step k
V  = zeros(x1_N, x2_N, N+1);      % value function
pi_star = zeros(x1_N, x2_N, N);   % policy (store optimal u at each state/time)

% terminal cost is the value function
for i = 1:x1_N
    for j = 1:x2_N

        % state at grid point
        x = [x1_grid(i); x2_grid(j)];

        % compute terminal cost
        V(i,j,N+1) = c_terminal(x);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BACKWARD DP (Bellman recursion)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf("Running backward DP...\n");

% backward recursion
for k = N:-1:1

    % loop over all grid states
    for i = 1:x1_N
        for j = 1:x2_N

            % state at grid point
            x = [x1_grid(i); x2_grid(j)];

            % initialize best cost and action
            J_star = inf;
            u_star = 0.0;
            
            % minimize over all discrete actions
            for a = 1:u_N

                % select action
                u = u_grid(a);

                % step dynamics
                x_next = f_disc(x, u);
                x_next(1) = wrap_to_pi(x_next(1));  % keep angle sane

                % get grid index for next state (snap to nearest grid point)
                [i_grid, j_grid] = nearest_grid_index(x_next, x1_grid, x2_grid);

                % Bellman backup:
                J = c_stage(x, u) + V(i_grid, j_grid, k+1);

                % check if this is the best so far
                if J < J_star
                    J_star = J;
                    u_star = u;
                end
            end

            % store best cost and action
            V(i,j,k) = J_star;
            pi_star(i,j,k) = u_star;  
        end
    end

    % progress update
    if mod(k,10)==0
        fprintf("  k = %d / %d\n", k, N);
    end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SAVE RESULTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% variables to save
vars = {'V', 'pi_star', 'x1_grid', 'x2_grid', 'u_grid', 'dt', 'N'};

% save the value function and policy
save('results.mat', vars{:});

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Helper Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% dynamics function
function [f_cont, f_disc] = make_dynamics_functions(params)

    % make symbolic variables
    syms x1 x2 u real

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

% cost functions
function [c_stage, c_terminal] = make_cost_functions(params, x_des)

    % extract cost parameters
    Q  = params.Q;
    R  = params.R;
    Qf = params.Qf;

    % cost function handles
    c_stage = @(x,u) 0.5 * (err(x,x_des)'*Q*err(x,x_des)) + 0.5*R*u*u;
    c_terminal = @(x) 0.5 * (err(x,x_des)'*Qf*err(x,x_des));
end

% error function (with angle wrapping)
function e = err(x, x_des)
    e1 = wrap_to_pi(x(1) - x_des(1));   % wrap the ANGLE ERROR
    e2 = x(2) - x_des(2);
    e = [e1; e2];
end

% wrap angle to [-pi, pi]
function angle_wrapped = wrap_to_pi(angle)
    angle_wrapped = mod(angle + pi, 2 * pi) - pi;
end

% find nearest grid index for a given state
function [i,j] = nearest_grid_index(x, x1_grid, x2_grid)
    % wrap angle and clamp omega to bounds
    x(1) = wrap_to_pi(x(1));
    x(2) = min(max(x(2), x2_grid(1)), x2_grid(end));

    % find nearest indices
    [~, i] = min(abs(x1_grid - x(1)));
    [~, j] = min(abs(x2_grid - x(2)));
end
