%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% KL Div between two different coin tosses
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc;

% Distribuutions
P1 = linspace(0.01, 0.99, 100);
P2 = 1 - P1;
P = [P1; P2];

Q1 = linspace(0.01, 0.99, 100);
Q2 = 1 - Q1;
Q = [Q1; Q2];

% create a matrix to store KL div
KL = zeros(100, 100);
for i = 1:100
    for j = 1:100
        P_ = P(:, i);
        Q_ = Q(:, j);
        KL(i, j) = KL_Divergence(P_, Q_);
    end
end

% plot the surface
figure;
axis equal; grid on;

% draw the surface
surf(P1, Q1, KL);
xlabel('p1');
ylabel('q1');
zlabel('KL divergence');

% function that compute Kl divergence given two distributions
function KL = KL_Divergence(P, Q)
    KL = 0;
    for i = 1:length(P)
        KL_ = P(i) * log(P(i)/Q(i));
        KL = KL + KL_;
    end
end


