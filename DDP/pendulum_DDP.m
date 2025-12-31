%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Differntial Dynamic Programming (DDP) for Pendulum Swing-Up
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; clc; close all;

% params
params.m = 1;
params.l = 1;
params.g = -9.81;

x0    = [0; 0];
x_des = [pi/2; 0];

tf = 8;
dt = 1/30;
tspan = 0:dt:tf;
N = numel(tspan)-1;

% Cost weights (tune these)
Q  = diag([50, 1]);
R  = 0.1;
Qf = diag([200, 20]);

umax = 10; % torque limit (optional but recommended)

% initial control guess (open-loop sequence)
U = zeros(N,1);

% DDP settings
maxIter = 60;
lambda  = 1e-3;     % regularization
lamMult = 10;
alphas  = [1, 0.5, 0.25, 0.1, 0.05];

% Rollout initial
[X, J] = rollout_pendulum(x0, U, x_des, params, dt, Q, R, Qf, umax);

for iter = 1:maxIter
    % Precompute derivatives along trajectory
    A = zeros(2,2,N);
    B = zeros(2,1,N);
    fxx_wrt_thth = zeros(N,1); % scalar for d2 omega_next / dtheta^2

    for k = 1:N
        th = X(1,k);
        [A(:,:,k), B(:,:,k), fxx_wrt_thth(k)] = dyn_derivs(th, params, dt);
    end

    % Backward pass
    Vx  = terminal_grad(X(:,N+1), x_des, Qf);
    Vxx = Qf;

    kff = zeros(1,N);      % feedforward term (scalar control)
    Kfb = zeros(1,2,N);    % feedback gain row vector

    diverged = false;

    for k = N:-1:1
        xk = X(:,k);
        uk = U(k);

        % Cost derivatives
        [lx, lu, lxx, luu, lux] = stage_cost_derivs(xk, uk, x_des, Q, R);

        Ak = A(:,:,k);
        Bk = B(:,:,k);

        % Q-function expansion (DDP)
        Qx  = lx  + Ak.' * Vx;
        Qu  = lu  + Bk.' * Vx;
        Qxx = lxx + Ak.' * Vxx * Ak;
        Quu = luu + Bk.' * Vxx * Bk;
        Qux = lux + Bk.' * Vxx * Ak;

        % --- DDP second-order dynamics contribution: sum_i Vx(i)*fxx_i
        % Only omega_next has nonzero second derivative wrt theta-theta:
        % Add to Qxx(1,1): Vx(2) * d2(omega_next)/dtheta^2
        Qxx(1,1) = Qxx(1,1) + Vx(2) * fxx_wrt_thth(k);

        % Regularize Quu
        Quu_reg = Quu + lambda;

        if Quu_reg <= 0
            diverged = true;
            break;
        end

        invQuu = 1 / Quu_reg;

        % Gains
        k_k = -invQuu * Qu;      % scalar
        K_k = -invQuu * Qux;     % 1x2

        kff(k)   = k_k;
        Kfb(:,:,k) = K_k;

        % Value function update
        Vx  = Qx  + K_k.' * (Quu * k_k + Qu) + Qux.' * k_k;
        Vxx = Qxx + K_k.' * (Quu * K_k) + K_k.' * Qux + Qux.' * K_k;
        Vxx = 0.5*(Vxx + Vxx.'); % symmetrize
    end

    if diverged
        lambda = lambda * lamMult;
        continue;
    end

    % Forward line search
    accepted = false;
    for a = alphas
        [Xnew, Jnew, Unew] = rollout_with_policy( ...
            x0, X, U, kff, Kfb, a, x_des, params, dt, Q, R, Qf, umax);

        if Jnew < J
            accepted = true;
            X = Xnew;
            U = Unew;
            J = Jnew;
            break;
        end
    end


    if accepted
        lambda = max(lambda / lamMult, 1e-6);
        fprintf('iter %d, cost %.6f, lambda %.2e\n', iter, J, lambda);
    else
        lambda = lambda * lamMult;
    end
end

% Plot
figure;
subplot(2,1,1); plot(tspan, X(1,:)); ylabel('\theta');
subplot(2,1,2); plot(tspan, X(2,:)); ylabel('\omega'); xlabel('t');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Helpers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [X, J] = rollout_pendulum(x0, U, x_des, params, dt, Q, R, Qf, umax)
    N = numel(U);
    X = zeros(2, N+1);
    X(:,1) = x0;
    J = 0;

    for k = 1:N
        uk = clamp(U(k), -umax, umax);
        xk = X(:,k);
        J = J + stage_cost(xk, uk, x_des, Q, R);
        X(:,k+1) = pendulum_step(xk, uk, params, dt);
    end
    J = J + terminal_cost(X(:,N+1), x_des, Qf);
end

function [Xnew, Jnew, Unew] = rollout_with_policy(x0, Xnom, Unom, kff, Kfb, alpha, x_des, params, dt, Q, R, Qf, umax)
    N = numel(Unom);
    Xnew = zeros(2,N+1);
    Unew = zeros(N,1);
    Xnew(:,1) = x0;
    Jnew = 0;

    for k = 1:N
        dx = Xnew(:,k) - Xnom(:,k);
        du = alpha*kff(k) + squeeze(Kfb(:,:,k)) * dx;
        uk = clamp(Unom(k) + du, -umax, umax);

        Unew(k) = uk;
        Jnew = Jnew + stage_cost(Xnew(:,k), uk, x_des, Q, R);
        Xnew(:,k+1) = pendulum_step(Xnew(:,k), uk, params, dt);
    end
    Jnew = Jnew + terminal_cost(Xnew(:,N+1), x_des, Qf);
end

function xnext = pendulum_step(x, u, params, dt)
    th = x(1); w = x(2);
    m = params.m; l = params.l; g = params.g;

    wdot = (g/l)*sin(th) + (1/(m*l^2))*u;
    th_next = th + dt*w;
    w_next  = w  + dt*wdot;

    xnext = [th_next; w_next];
end

function [A, B, fxx_thth] = dyn_derivs(theta, params, dt)
    m = params.m; l = params.l; g = params.g;
    c = g/l;
    b = 1/(m*l^2);

    A = [1, dt;
         dt*c*cos(theta), 1];

    B = [0;
         dt*b];

    % d^2 omega_next / dtheta^2 = dt * (g/l) * (-sin(theta))
    fxx_thth = dt * c * (-sin(theta));
end

function c = stage_cost(x, u, x_des, Q, R)
    e = x - x_des;
    c = 0.5 * (e.'*Q*e) + 0.5 * (u.'*R*u);
end

function c = terminal_cost(x, x_des, Qf)
    e = x - x_des;
    c = 0.5 * (e.'*Qf*e);
end

function [lx, lu, lxx, luu, lux] = stage_cost_derivs(x, u, x_des, Q, R)
    e = x - x_des;
    lx  = Q*e;
    lu  = R*u;
    lxx = Q;
    luu = R;
    lux = zeros(1,2);
end

function Vx = terminal_grad(x, x_des, Qf)
    e = x - x_des;
    Vx = Qf*e;
end

function y = clamp(x, lo, hi)
    y = min(max(x, lo), hi);
end
