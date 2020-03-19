% get_cost: Function constructing Hessian of cost function
%
% Inputs:
%       r:              Non-negative scalar penalizing input
%       Q:              Symmetric positive semidefinite matrix penalizing
%                       errors in state
%       n_lookahead:    Length of MPC prediction horizon
%
% Outputs:
%       S:              Hessian matrix of cost function
%
% --
% Control for Robotics
% AER1517 Spring 2020
% Programming Exercise 2
%
% --
% University of Toronto Institute for Aerospace Studies
% Dynamic Systems Lab
%
% Course Instructor:
% Angela Schoellig
% schoellig@utias.utoronto.ca
%
% Teaching Assistant:
% SiQi Zhou
% siqi.zhou@robotics.utias.utoronto.ca
%
% --
% Revision history
% [20.03.07, SZ]    first version

function [S] = get_cost(r, Q, n_lookahead)
    % Consutruct Hessian matrix
    S = [];
    for i = 1:1:n_lookahead
        S = blkdiag(S, r);
    end

    for i = 1:1:n_lookahead
        S = blkdiag(S, Q);
    end
end