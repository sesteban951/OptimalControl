%  draw the cartpole given the state x
function [pole_base, pole, ball, cart] = draw_cartpole(x, params)

    % parameters
    l = params.l; % length of the pole

    % colors
    gray = [0.5, 0.5, 0.5];
    orange = [1, 0.5, 0];

    % position of the cartpole
    th = x(1);
    p = x(2);

    % cart and pole positions
    cart_pos = [p; 0];
    pole_pos = [p + l*sin(th); -l*cos(th)];

    % draw the cart
    cart = rectangle('Position', [cart_pos(1)-0.1, cart_pos(2)-0.05, 0.2, 0.1], 'Curvature', 0.1, 'FaceColor', gray);
    pole_base = plot(cart_pos(1), cart_pos(2), 'ko', 'LineWidth', 2, 'MarkerFaceColor', 'k');

    % draw pole
    pole = plot([cart_pos(1), pole_pos(1)], [cart_pos(2), pole_pos(2)], 'k', 'LineWidth', 2);
    ball = plot(pole_pos(1), pole_pos(2), 'ko', 'MarkerSize', 16, 'MarkerFaceColor', orange);

end