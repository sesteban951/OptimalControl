clear all; clc; close all;

% inverted pendulum parameters
params.m = 1; % mass of the pendulum
params.l = 1; % length of the pendulum
params.g = -9.81; % gravity

% initial condition
x0 = [0; 0];
x_des = [pi/2; 0];

% time span
tf = 8;
t0 = 0;
dt = 1/30;
tspan = t0 : dt : tf;

% solve the ODE
global u_list;
u_list = [];
[t, x] = ode45(@(t, x) inverted_pendulum(t, x, x_des, params), tspan, x0);

figure(1);
axis equal;
xlim = [-params.l-0.5, params.l+0.5];
ylim = [-params.l-0.5, params.l+0.5];
axis([xlim, ylim]);
pause(0.5);
tic;
for i = 1:length(t)

    while toc < t(i)
        % wait
    end

    title(sprintf('Time: %.2f', t(i)));

    % draw the pendulum
    draw_pendulum(t, x(i, :), params);

    % clear the plot
    if i ~= length(t)
        delete(findobj('type', 'line'));
        delete(findobj('type', 'rectangle'));
    end
end

% plot the result
figure(2);
subplot(1, 2, 1);
plot(t, x);
ylabel('State');
xlabel('Time (s)');
legend('theta', 'theta dot');
grid on;

subplot(1, 2, 2);
plot(t, u_list);
ylabel('Control input');
xlabel('Time (s)');
legend('u');
grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% dynamic of the pendulum
function xdot = inverted_pendulum(t, x, x_des, params)

    theta = x(1);     % position 
    theta_dot = x(2); % velocity

    m = params.m; % mass of the pendulum
    l = params.l; % length of the pendulum
    g = params.g; % gravity

    % choose control policy
    u = k(t, x, x_des, params);
    global u_list;
    u_list = [u_list; u];

    % dynamics of the pendulum
    f_x = [theta_dot; 
           (g/l) * sin(theta)];
    g_x = [0; 
           1/(m*l^2)];

    xdot = f_x + g_x * u;

end

% control policy
function u = k(t, x, x_des, params)

    theta = x(1);             % position 
    theta_dot = x(2);         % velocity
    theta_des = x_des(1);     % desired position
    theta_dot_des = x_des(2); % desired velocity

    % simple PD for now
    Kp = 20;
    Kd = 5;
    u = -Kp * (theta - theta_des) - Kd * (theta_dot - theta_dot_des);

end
    


