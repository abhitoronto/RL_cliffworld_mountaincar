% get_lin_matrices: Function computing the Jacobian matrices for the
%                   mountain car dynamics model
%
% Inputs:
%       (state, input): operating point which the system dynamics is
%                       linearized about
%
% Outputs:
%       A:              Jacobian of dynamics with respect to state 
%                       evaluated at (state, input)
%       B:              Jacobian of dynamics with respect to input 
%                       evaluated at (state, input)
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

function [A,B] = get_lin_matrices(state, input)
    % A
    a11 = 1+0.0075*sin(3*state(1));
    a12 = 1;
    a21 = 0.0075*sin(3*state(1));
    a22 = 1; 
    A = [a11, a12; a21, a22];
    
    % B
    B = [0.001; 0.001];
end

