% dyncons: Function defninig nonlinear system dynamics constraints 
%          (used in fmincon)
%
% Inputs:
%       x:              A vector of decision variables 
%                       [input,...,input, state', ..., state']^T
%
% Outputs:
%       c:              Nonlinear inequality constraints
%       ceq:            Nonlinear equality constraints
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

function [c,ceq] = dyncons(x)
    % Load parameters and current state
    load('params');
    load('cur_state');

    % Extract input and state from the vector x
    INPUTS = x(1:n_lookahead*dim_action);
    STATES_CROSSTERMS = x(n_lookahead*dim_action+1:end);
    odd_idx = 1:2:2*n_lookahead;
    even_idx = odd_idx + 1;
    POSITIONS = STATES_CROSSTERMS(odd_idx);
    VELOCITIES = STATES_CROSSTERMS(even_idx);
    
    % Define equality constraints
    const_counter = 0;
    for i = 1:1:n_lookahead
        if i == 1
            const_counter = const_counter + 1;
            ceq(const_counter) = -VELOCITIES(i) + cur_state(2) ...
                + 0.001*INPUTS(i) - 0.0025*cos(3*cur_state(1));
            
            const_counter = const_counter + 1;
            ceq(const_counter) = -POSITIONS(i) + cur_state(1) ...
                + VELOCITIES(i);
        else
            const_counter = const_counter + 1;
            ceq(const_counter) = -VELOCITIES(i) + VELOCITIES(i-1) + ...
                0.001*INPUTS(i) - 0.0025*cos(3*POSITIONS(i-1));
            
            const_counter = const_counter + 1;
            ceq(const_counter) = -POSITIONS(i) + POSITIONS(i-1) + ...
                VELOCITIES(i);
        end
    end 
    
    % Inequality constraints
    c = [];
end

