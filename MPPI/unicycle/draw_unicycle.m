function draw_unicycle(t, x, fig, xlim, ylim)

    % position of the unicycle
    y = x(1);
    z = x(2);
    theta = x(3) - pi/2;

    % draw the unicycle as polyhedron
    b = 0.15;
    h = 0.25;

    % vertices of the unicycle
    y_vert = [0.5 * b, 0, -0.5 * b];
    z_vert = [-(1/3)*h, (2/3)*h, -(1/3)*h];
    R = [cos(theta), -sin(theta); 
         sin(theta), cos(theta)];
    verts = [y_vert; z_vert];
    vertices = R * verts;
    y_vert = vertices(1, :) + y;
    z_vert = vertices(2, :) + z;

    % plot the unicycle
    figure(fig);
    hold on;
    patch('Vertices', [y_vert', z_vert'], 'Faces', [1, 2, 3], 'FaceColor', 'c'); 
    plot(y,z, "MarkerSize", 5, "Marker", "o", "MarkerFaceColor", "k"); 
    hold off;
    title(sprintf('Time: %.2f', t));
    axis([xlim, ylim]);
    grid on; axis equal;

    drawnow;

