% build_stochastic_mdp_nn: Function implementing the Nearest Neighbour
%                          approach for creating a stochastic MDP
%
% Inputs:
%       world:                  A structure containing basic parameters for
%                               the mountain car problem
%       T:                      Transition model with elements initialized
%                               to zero
%       R:                      Expected reward model with elements
%                               initialized to zero
%       num_samples:            Number of samples to use for creating the
%                               stochastic model
%
% Outputs:
%       T:                      Transition model with elements T{a}(s,s')
%                               being the probability of transition to 
%                               state s' from state s taking action a
%       R:                      Expected reward model with elements 
%                               R{a}(s,s') being the expected reward on 
%                               transition from s to s' under action a
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

function [T, R] = build_stochastic_mdp_nn(world, T, R, num_samples)
    % Extract states and actions
    STATES = world.mdp.STATES;
    ACTIONS = world.mdp.ACTIONS;

    % Dimensions
    num_states = size(STATES, 2);
    num_actions = size(ACTIONS, 2);
    
    % Noise Parameters
    var_p = 0.14/40;
%     var_p = 0.0;
    std_state = [var_p, var_p];
    
    % Value Bounds
    pos_bounds = world.param.pos_bounds;
    vel_bounds = world.param.vel_bounds;
    
    % Goal
    s_goal = world.param.s_goal;
    s_goal_indices = [];
    for it = 1:size(s_goal,2)
        s_goal_indices = [s_goal_indices, ...
            nearest_state_index_lookup(STATES, s_goal(:, it))];
    end
    
    % Misc
    visits = {};
    for action_index = 1:1:num_actions
        visits{action_index} = zeros(num_states, num_states);
    end

    % Loop through all possible states
    for state_index = 1:1:num_states
        cur_state = STATES(:, state_index);
        x = cur_state(1);
        v = cur_state(2);
        fprintf('building model... state %d\n', state_index);

        % Apply each possible action
        for action_index = 1:1:num_actions
            action = ACTIONS(:, action_index);

            % Build a stochastic MDP based on Nearest Neighbour
            % Note: The function 'nearest_state_index_lookup' can be used
            % to find the nearest node to a countinuous state
            for samples = 1:1:num_samples
                                
                % Get next state index
                [~,next_state,reward,~] = one_step_mc_model_noisy(world, cur_state, action, std_state);
                next_state_index = nearest_state_index_lookup(STATES, next_state);
                
                % Update transition and reward models
                T{action_index}(state_index, next_state_index) = 1/num_samples + ...
                    T{action_index}(state_index, next_state_index);
                
                R{action_index}(state_index, next_state_index) = reward + ...
                    R{action_index}(state_index, next_state_index);
                visits{action_index}(state_index, next_state_index) = 1 + ...
                    visits{action_index}(state_index, next_state_index);

            end
        end
    end
    for action_index = 1:1:num_actions
        non_zero = find(R{action_index} ~= 0.0);
        R{action_index}(non_zero) = R{action_index}(non_zero)./...
            visits{action_index}(non_zero);
    end
end

