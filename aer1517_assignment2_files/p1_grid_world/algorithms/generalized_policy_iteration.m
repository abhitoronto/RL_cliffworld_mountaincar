% generalized_policy_iteration: Function solving the given MDP using the
%                               Generalized Policy Iteration algorithm
%
% Inputs:
%       world:                  A structure defining the MDP to be solved
%       precision_pi:           Maximum value function change before
%                               terminating Policy Improvement step
%       max_ite_pi:             Maximum number of iterations for Policy
%                               Improvement loop
%       precision_pe:           Maximum value function change before
%                               terminating Policy Evaluation step
%       max_ite_pe:             Maximum number of iterations for Policy
%                               Evaluation loop
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
% This script is adapted from the course on Optimal & Learning Control for
% Autonomous Robots at the Swiss Federal Institute of Technology in Zurich
% (ETH Zurich). Course Instructor: Jonas Buchli. Course Webpage:
% http://www.adrlab.org/doku.php/adrl:education:lecture:fs2015
%
% --
% Revision history
% [20.03.07, SZ]    first version

function [V, policy_index] = generalized_policy_iteration(world, precision_pi, precision_pe, max_ite_pi, max_ite_pe)
    %% Initialization
    % MDP
    mdp = world.mdp;
    T = mdp.T; % transition_probability
    R = mdp.R; % Reward function 
    gamma = mdp.gamma;

    % Dimensions
    num_actions = length(T);
    num_states = size(T{1}, 1);

    % Intialize value function
    V = zeros(num_states, 1);

    % Initialize policy
    % Note: Policy here encodes the action to be executed at state s. We
    %       use deterministic policy here (e.g., [0,1,0,0] means take 
    %       action indexed 2)
    random_act_index = randi(num_actions, [num_states, 1]);
    policy = zeros(num_states, num_actions);
    for s = 1:1:num_states
        selected_action = random_act_index(s);
        policy(s, selected_action) = 1;
    end
    
    fprintf('\n\n\t########### Policy Iteration ########\n')
    
    policy_last = [];
    for  i = 1:max_ite_pe
        policy_last = policy;
        
        %% [TODO] policy Evaluation (PE) (Section 2.6 of [1])
        % V = ...;
        V_last = [];
        
        for j = 1:max_ite_pi
            V_last = V;
            Q = zeros(num_states, num_actions);
            for ai = 1:1:num_actions
                Q(:, ai) = diag(T{ai}*transpose(R{ai})) + gamma*T{ai}*V;
            end
            V = sum(policy.*Q, 2);
            V_error = norm(V-V_last);
            if V_error < precision_pi
                disp(num2str(i) + ": " + num2str(j) +  ": Policy Iteration Converged: " + num2str(V_error));
                break;
            end
        end
        
        %% [TODO] Policy Improvment (PI) (Section 2.7 of [1])
        % policy = ...;
        [temp, policy_index] = max(Q,[],2);
        policy = zeros(num_states, num_actions);
        for s = 1:1:num_states
            selected_action = policy_index(s);
            policy(s, selected_action) = 1;
        end

        policy_error = norm(policy-policy_last);
        % Check algorithm convergence
        if policy_error < precision_pe
            disp(num2str(i) + ": Policy Evaluation Converged: " + num2str(policy_error));
            break;
        end
    end
    
    % Return deterministic policy for plotting
    [~, policy_index] = max(policy, [], 2);
end