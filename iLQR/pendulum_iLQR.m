%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Iterative LQR (iLQR) for Pendulum Swing-Up
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; clc; close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SETUP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% simulation parameters
tf = 8.0;
dt = 0.025;
tspan = 0:dt:tf;
N = numel(tspan) - 1;

% create the dynamics function
dyn_params.m = 1;
dyn_params.b = 0.01;
dyn_params.l = 1;
dyn_params.g = 9.81;
dyn_params.dt = dt;

% create the cost function 
cost_params.Q  = diag([10, 10, 1.0]);    
cost_params.Qf = diag([100, 100, 10]);
cost_params.R = 0.01;           

% set optimization options
opt_params.max_iters = 500;                                % maximum iLQR iterations
opt_params.tol       = 1e-6;                               % convergence tolerance
opt_params.alphas    = [1.0, 0.75, 0.5, 0.25, 0.1, 0.01];  % line search alphas
opt_params.mu0       = 1e-6;                               % initial regularization for Quu
opt_params.mu_factor = 10;                                 % mu increase/decrease factor
opt_params.mu_min    = 1e-6;                               % minimum mu
opt_params.mu_max    = 1e3;                                % maximum mu

% function handle for discrete dynamics with Jacobians (consistent RK4)
% f_step(x,u) -> [x_next, Ad, Bd]
f_step = @(x,u) rk4_step_with_jacobians(x, u, dyn_params);

% plotting options
plot_setting = 1; % 1: pendulum, 2: normal plots

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MAIN iLQR LOOP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% initial condition
x0 = [pi/2; -1.0];

% initial guesses
xdes = [pi; 0];               % desired upright state
U    = zeros(1, N) + 0.01;   % initial control guess (1xN)

% do an initial rollout, TODO: make a better initial guess like energy shaping
[X, J] = rollout(x0, xdes, U, f_step, cost_params, N);

% history
J_hist = zeros(opt_params.max_iters+1,1);
J_hist(1) = J;

% initialize regularization parameter
mu = opt_params.mu0;

% main iLQR iteration
fprintf("Starting iLQR optimization...\n");
for iter = 1:opt_params.max_iters

    % linearize along current trajectory
    [Ad_list, Bd_list] = linearize_about_trajectory(X, U, f_step, N);

    % backward pass (if fails, increase mu and retry)
    [k_seq, K_seq, success] = backward_pass(X, U, xdes, Ad_list, Bd_list, N, mu, cost_params);

    % check if backward pass succeeded
    if success == false
        % increase mu and try again
        mu = min(mu * opt_params.mu_factor, opt_params.mu_max);
        fprintf("iter %d: backward pass failed, increasing mu -> %.2e\n", iter, mu);

        % if mu hits max and still failing, bail
        if mu >= opt_params.mu_max
            warning("mu hit mu_max=%.2e and backward pass still failing. Stopping.", opt_params.mu_max);
            J_hist = J_hist(1:iter);
            break;
        end
        continue;
    end

    % forward pass line search
    accepted = false;                                             % flag for accepted step
    best_step = struct('J', inf, 'X', [], 'U', [], 'alpha', NaN); % best found step

    % try different alphas
    for a = opt_params.alphas

        % perform forward pass with this alpha
        [X_try, U_try, J_try] = forward_pass(x0, xdes, X, U, k_seq, K_seq, a, f_step, cost_params, N);
        
        % check if this is the best step so far and store it
        if J_try < best_step.J
            best_step.J = J_try;
            best_step.X = X_try;
            best_step.U = U_try;
            best_step.alpha = a;
        end

        % accept first improving step
        if J_try < J
            accepted = true;
            break;
        end
    end

    % update trajectories if step was accepted
    if accepted == true

        % update trajectories
        X = best_step.X;
        U = best_step.U;

        % compute cost improvement
        dJ = J - best_step.J;
        J  = best_step.J;
        J_hist(iter+1) = J;

        % decrease mu when step succeeds (trust model more)
        mu = max(mu / opt_params.mu_factor, opt_params.mu_min);

        fprintf("iter %d: J=%.3f, dJ=%.3e, alpha=%.3f, mu=%.2e\n", ...
                iter, J, dJ, best_step.alpha, mu);

        % convergence check
        if abs(dJ) < opt_params.tol
            J_hist = J_hist(1:iter+1);
            break;
        end

    % no acceptable step found, increase mu
    else
        % no alpha improved cost -> increase mu and try again
        mu = min(mu * opt_params.mu_factor, opt_params.mu_max);
        fprintf("iter %d: no improvement, increasing mu -> %.2e\n", iter, mu);

        if mu >= opt_params.mu_max
            warning("mu hit mu_max=%.2e and still no improvement. Stopping.", opt_params.mu_max);
            J_hist = J_hist(1:iter+1);
            break;
        end
    end
end

fprintf("iLQR optimization complete in %d iterations.\n", size(J_hist,1)-1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PLOTTING
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% animate the pendulum
if plot_setting == 1

    % import helper functions
    addpath('../InvertedPendulum/');
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
        [base, pole, ball] = draw_pendulum(time, X(:,k), dyn_params);

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

% plot all sorts of stuff
if plot_setting == 2
    
    figure(2);

    % pendulum angle
    subplot(2,2,1);
    plot(tspan, X(1,:), 'LineWidth', 2);
    xlabel('Time (s)');
    ylabel('Angle (rad)');
    title('Pendulum Angle Trajectory');
    grid on;

    % pendulum angular velocity
    subplot(2,2,2);
    plot(tspan, X(2,:), 'LineWidth', 2);
    xlabel('Time (s)');
    ylabel('Angular Velocity (rad/s)');
    title('Pendulum Angular Velocity Trajectory');
    grid on;

    % plot control inputs
    subplot(2,2,3);
    stairs(tspan(1:end-1), U, 'LineWidth', 2);
    xlabel('Time (s)');
    ylabel('Control Input (Torque)');
    title('Control Input Trajectory');
    grid on;

    % plot cost history
    subplot(2,2,4);
    plot(0:length(J_hist)-1, J_hist, '-o', 'LineWidth', 2);
    xlabel('iLQR Iteration');
    ylabel('Total Cost J');
    title('iLQR Cost History');
    grid on;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% iLQR Helper Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% stage cost function and
function [l, lx, lu, lxx, luu, lux] = stage_cost(x, u, xdes, cost_params)

    % unpack cost parameters
    Q = cost_params.Q;   % 3x3
    R  = cost_params.R;    % scalar

    % unpack state
    th = x(1);  
    w = x(2);
    th_des = xdes(1);
    w_des = xdes(2);

    % compute the error
    y    = [cos(th); 
            sin(th); 
            w];
    ydes = [cos(th_des); 
            sin(th_des); 
            w_des];
    e = y - ydes;

    % output Jacobian
    J = [-sin(th)  0;
          cos(th)  0;
          0        1];

    % compute the stage cost
    l  = 0.5 * (e.' * Q * e) + 0.5 * (u.' * R * u);
    lx  = J.' * (Q * e);
    lu  = R * u;

    % compute the Hessians
    lxx = J.' * Q * J;  % approximated via Gauss-Newton Hessian (PSD)
    luu = R;
    lux = zeros(1,2);
end

% terminal cost function
function [lf, lfx, lfxx] = terminal_cost(x, xdes, cost_params)

    % unpack cost parameters
    Qf = cost_params.Qf; % 3x3

    % unpack state
    th = x(1);  
    w = x(2);
    th_des = xdes(1);
    w_des = xdes(2);

    % compute the error
    y    = [cos(th); 
            sin(th); 
            w];
    ydes = [cos(th_des); 
            sin(th_des); 
            w_des];
    e = y - ydes;

    % output Jacobian
    J = [-sin(th)  0;
          cos(th)  0;
          0        1];

    % compute the terminal cost
    lf = 0.5 * (e.' * Qf * e); 
    lfx  = J.' * (Qf * e); 

    % compute the Hessian
    lfxx = J.' * Qf * J;      % approximated via Gauss-Newton Hessian (PSD)
end

% forward pass: apply du = alpha * k_ff + k_fb * (x_new - x_nom) and roll out
function [X_new, U_new, J_new] = forward_pass(x0, xdes,X_nom, U_nom, ...
                                              K_ff_seq, K_fb_seq, ...
                                              alpha, f_step, cost_params, N)

    % pre allocate new trajectories
    X_new = zeros(2, N+1);
    U_new = zeros(1, N);
    X_new(:,1) = x0;

    % initialize new cost
    J_new = 0;

    % rollout with updated controls
    for k = 1:N
        
        % current new state and nominal state
        xk_new = X_new(:,k);
        xk_nom = X_nom(:,k);

        % feedback gain at time k (1x2)
        k_ff = K_ff_seq(k);               % -> scalar
        k_fb = squeeze(K_fb_seq(:,:,k));  % -> 1x2

        % control update
        dx = xk_new - xk_nom;           % (2x1)
        dx(1) = wrap_pi(dx(1));         % wrap angle error
        du = alpha * k_ff + k_fb * dx;  % scalar

        % updated control
        uk_new = U_nom(k) + du;
        
        % enforce control limits
        % umax = 7.0;
        % uk_new = max(min(uk_new, umax), -umax);
        
        % stage cost
        [lk, ~, ~, ~, ~, ~] = stage_cost(xk_new, uk_new, xdes, cost_params);
        J_new = J_new + lk;
        
        % propagate nonlinear dynamics (ignore A,B here)
        [xk_next, ~, ~] = f_step(xk_new, uk_new);

        % store results
        X_new(:,k+1) = xk_next;
        U_new(k) = uk_new;
    end

    % terminal cost
    [lf, ~, ~] = terminal_cost(X_new(:,end), xdes, cost_params);
    J_new = J_new + lf;
end

% backward pass: compute optimal control law
function [K_ff_seq, K_fb_seq, success] = backward_pass(X, U, xdes, ...
                                                       Ad_list, Bd_list, ...
                                                       N, mu, cost_params)

    % create storage for gains
    K_ff_seq = zeros(1, N);       % feedforward
    K_fb_seq = zeros(1, 2, N);    % feedback
    success = true;               % flag for successful backward pass

    % terminal value function from terminal cost
    [~, Vx, Vxx] = terminal_cost(X(:,end), xdes, cost_params);

    % backwards dynamic programming
    for k = N:-1:1

        % get current state and input
        xk = X(:,k);
        uk = U(k);

        % get linearized dynamics at (xk, uk)
        Ad = Ad_list(:,:,k);
        Bd = Bd_list(:,:,k);

        % stage cost derivatives at (xk, uk)
        [~, lx, lu, lxx, luu, lux] = stage_cost(xk, uk, xdes, cost_params);

        % Q-function derivatives
        % Q = l(x,u) + V(f(x,u))
        % Q ≈ l + V + Vx' (Ad dx + Bd du) + 0.5 [dx;du]' [Ad' Bd'] Vxx [Ad Bd] [dx;du]
        Qx  = lx  + Ad' * Vx;
        Qu  = lu  + Bd' * Vx;
        Qxx = lxx + Ad' * Vxx * Ad;
        Quu = luu + Bd' * Vxx * Bd;
        Qux = lux + Bd' * Vxx * Ad;

        % regularize Quu to keep it PD / invertible
        Quu_reg = Quu + mu;          % always regularize
        if Quu_reg <= 0
            success = false;
            return;
        end

        %  local optimal control law
        % du = argmin (0.5 du' Quu du + (Qux dx + Qu)' du + const)
        K_ff = -(Quu_reg \ Qu);      % (1x1)
        K_fb = -(Quu_reg \ Qux);     % (1x2)
        K_ff_seq(:,k)   = K_ff;
        K_fb_seq(:,:,k) = K_fb;

        % update value function approximation
        Vx  = Qx  + K_fb' * Quu_reg * K_ff + K_fb' * Qu + Qux'  * K_ff;
        Vxx = Qxx + K_fb' * Quu_reg * K_fb + K_fb' * Qux + Qux' * K_fb;

        % symmetrize for numerical stability
        Vxx = 0.5 * (Vxx + Vxx');
    end
end

% dynamics rollout
function [X, J] = rollout(x0, xdes, U, f_step, cost_params, N)

    % pre allocate state trajectory
    X = zeros(2, N+1);
    X(:,1) = x0;

    % initialize cost
    J = 0;

    % rollout dynamics (using nonlinear dynamics with rk4)
    xk = x0;
    for k = 1:N

        % get the current state and input
        uk = U(k);

        % compute stage cost and accumulate
        [lk, ~, ~, ~, ~, ~] = stage_cost(xk, uk, xdes, cost_params);
        J = J + lk;

        % propagate dynamics
        [xk, ~, ~] = f_step(xk, uk);

        % store the state
        X(:,k+1) = xk;
    end

    % terminal cost
    [lf, ~, ~] = terminal_cost(xk, xdes, cost_params);
    J = J + lf;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Helper Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% continuous dynamics
function xdot = f_cont(x, u, params)

    % unpack the parameters
    m = params.m; 
    b = params.b; 
    l = params.l; 
    g = params.g;

    % dynamics
    xdot = [ x(2);
             -(g/l)*sin(x(1)) - (b/(m*l^2))*x(2) + (1/(m*l^2))*u ];
end

% continuous linear Ac matrix
function Ac = make_Ac(x, params)

    % unpack the parameters
    m = params.m; 
    b = params.b; 
    l = params.l; 
    g = params.g;

    % linearized drift term
    Ac = [ 0,                1;
          -(g/l)*cos(x(1)), -(b/(m*l^2)) ];
end

% continuous linear Bc matrix
function Bc = make_Bc(~, params)
    
    % unpack the parameters
    m = params.m; 
    l = params.l;

    % linearized control term
    Bc = [0;
          1/(m*l^2)];
end

% RK4 step + exact discrete-time linearization (consistent with the rollout)
%
% Given the continuous-time dynamics xdot = f_cont(x,u), this function:
%   1) Propagates the state forward one timestep dt using RK4:
%        x_{k+1} = f_d(x_k, u_k)
%   2) Simultaneously computes the discrete-time Jacobians of that SAME RK4 map:
%        Ad = ∂f_d/∂x  evaluated at (x_k,u_k)   (2x2)
%        Bd = ∂f_d/∂u  evaluated at (x_k,u_k)   (2x1)
%
% These Jacobians give the local discrete-time linear model used by iLQR:
%        δx_{k+1} ≈ Ad δx_k + Bd δu_k
%
% How it works (sensitivity propagation):
%   Along with x(t), we integrate two sensitivity matrices over the interval [0, dt]
%   assuming u is held constant during the step:
%
%     Phi(t)   = ∂x(t)/∂x(0)     with initial condition Phi(0)   = I
%     Gamma(t) = ∂x(t)/∂u        with initial condition Gamma(0) = 0
%
%   They satisfy the continuous-time variational equations:
%     d/dt Phi   = Ac(x) * Phi
%     d/dt Gamma = Ac(x) * Gamma + Bc(x)
%
%   where Ac(x) = ∂f/∂x and Bc(x) = ∂f/∂u for the CONTINUOUS dynamics.
%
% Output:
%   x_next : RK4 next state
%   Ad     : discrete-time Jacobian ∂x_next/∂x (matching RK4)
%   Bd     : discrete-time Jacobian ∂x_next/∂u (matching RK4)
function [x_next, Ad, Bd] = rk4_step_with_jacobians(x, u, params)

    % unpack parameters
    dt = params.dt;

    % Augmented dynamics RHS for (x, Phi, Gamma)
    % xdot      = f(x,u)                         (2x1)
    % Phidot    = Ac(x)*Phi                      (2x2)
    % Gammadot  = Ac(x)*Gamma + Bc(x)            (2x1)
    function [xdot, Phidot, Gammadot] = aug_dyn(x, u, Phi, Gamma)
        xdot = f_cont(x, u, params);
        Ac   = make_Ac(x, params);
        Bc   = make_Bc(x, params);
        Phidot   = Ac * Phi;
        Gammadot = Ac * Gamma + Bc;
    end

    % initialize sensitivities 
    Phi   = eye(2);
    Gamma = zeros(2,1);

    % RK4 integration
    [k1, P1, G1] = aug_dyn(x, u, Phi, Gamma);

    x2   = x   + 0.5*dt*k1;   
    Phi2 = Phi + 0.5*dt*P1;   
    Gam2 = Gamma + 0.5*dt*G1;
    [k2, P2, G2] = aug_dyn(x2, u, Phi2, Gam2);

    x3   = x   + 0.5*dt*k2;   
    Phi3 = Phi + 0.5*dt*P2;   
    Gam3 = Gamma + 0.5*dt*G2;
    [k3, P3, G3] = aug_dyn(x3, u, Phi3, Gam3);

    x4   = x   + dt*k3;       
    Phi4 = Phi + dt*P3;       
    Gam4 = Gamma + dt*G3;
    [k4, P4, G4] = aug_dyn(x4, u, Phi4, Gam4);

    % discrete update stuff
    x_next = x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4);
    Ad     = Phi + (dt/6)*(P1 + 2*P2 + 2*P3 + P4);
    Bd     = Gamma + (dt/6)*(G1 + 2*G2 + 2*G3 + G4);
end

% wrap angle to [-pi, pi]
function a = wrap_pi(a)
    a = mod(a + pi, 2*pi) - pi;
end

% linearize about a trajectory
function [Ad_list, Bd_list] = linearize_about_trajectory(X, U, f_step, N)

    % pre allocate linearization matrices
    Ad_list = zeros(2, 2, N);
    Bd_list = zeros(2, 1, N);

    % linearize at each timestep
    for k = 1:N

        % get the current state and input
        xk = X(:,k);
        uk = U(k);

        % take a step and get Jacobians
        [~, Ad, Bd] = f_step(xk, uk);

        % store the linearization
        Ad_list(:,:,k) = Ad;
        Bd_list(:,:,k) = Bd;
    end
end
