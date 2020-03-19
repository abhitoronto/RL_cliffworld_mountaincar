% linear_programming: Function solving the given MDP using the Linear
%                     Programming approach
%
% Inputs:
%       world:                  A structure defining the MDP to be solved
%
% Outputs:
%       V:                      An array containing the value at each state
%       policy_index:           An array summarizing the index of the
%                               optimal action index at each state
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

function [V, policy] = linear_programming(world)
    %% Initialization
    % MDP
    mdp = world.mdp;
    T = mdp.T; % transition_probability
    R = mdp.R; % Reward function 
    gamma = mdp.gamma;
    
    % Dimensions
    num_actions = length(T);
    num_states = size(T{1}, 1);

    
    fprintf('\n\n\t########### Linear Programming ########\n')
    
    %% [TODO] Compute optimal value function (see [2] for reference)
    % V = ...;
    
    f = ones(num_states,1);
    A = [];
    b = [];
    for action_index = 1:1:num_actions 
        A = [A; gamma*T{action_index} - eye(num_states)];
        b = [b; -diag(T{action_index}*R{action_index}')];
    end
    
    V = linprog(f, A, b);
    
    %% [TODO] Compute an optimal policy
    % policy = ...;
    Q = zeros(num_actions, num_states);
    for action_index = 1:1:num_actions
        Q(action_index, :) = diag(T{action_index}*R{action_index}') + gamma*T{action_index}*V;
    end
    
    [temp, policy] = max(Q,[],1);
end
