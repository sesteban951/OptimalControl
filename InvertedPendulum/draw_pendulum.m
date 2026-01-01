%  draw the pendulum given the state x
function [base, pole, ball] = draw_pendulum(t, x, params)

    % parameters
    l = params.l; % length of the pendulum
    m = params.m; % mass of the bob
    
    % position of the pendulum
    theta = x(1);
    
    % draw the base of the pendulum
    x_base = 0;
    y_base = 0;
    w_base = 0.1;
    h_base = 0.05;
    base = rectangle('Position', [x_base - w_base/2, y_base - h_base/2, w_base, h_base],...
                     'Curvature', [0.5, 0.5], ...
                     'FaceColor', 'k');

    % draw the pednulum rod
    x_rod = [x_base, x_base + l * sin(theta)];
    y_rod = [y_base, y_base - l * cos(theta)];
    pole = line(x_rod, y_rod, 'LineWidth', 2, 'Color', 'k');

    % draw the pendulum ball
    x_ball = x_base + l * sin(theta);
    y_ball = y_base - l * cos(theta);
    r_ball = 0.1 * sqrt(m);
    ball = rectangle('Position', [x_ball-r_ball, y_ball-r_ball, 2*r_ball, 2*r_ball],...
                     'Curvature', [1, 1], ...
                     'FaceColor', '[0.6350 0.0780 0.1840]');
end