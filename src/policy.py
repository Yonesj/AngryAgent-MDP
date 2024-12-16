from typing import List, Dict, Tuple, Set

from environment import AngryBirds
from const import actions, action_codes


def get_states_from_transitions(transitions: Dict[Tuple[int, int], Dict[int, List]]) -> Set[Tuple[int, int]]:
    if isinstance(transitions, dict):
        return set(transitions.keys())
    raise Exception('Could not retrieve states from transitions')


def expected_utility(
        action: Tuple[int, int],
        state: Tuple[int, int],
        utility: Dict[Tuple[int, int], int],
        mdp: AngryBirds) -> float:
    """The expected utility of doing a in state s, according to the MDP and U."""
    return sum([p * utility[s1] for (p, s1, r) in mdp.transition_table[state][action_codes[action]]])


def policy_evaluation(pi: Dict[Tuple[int, int], Tuple[int, int]],
                      utility: Dict[Tuple[int, int], int],
                      mdp: AngryBirds,
                      states: Set[Tuple[int, int]],
                      gamma: float,
                      # k=20
                      epsilon=1e-4
                      ) -> Dict[Tuple[int, int], int]:
    """Return an updated utility mapping U from each state in the MDP to its
    utility, using an approximation (modified policy iteration)."""
    R, T = mdp.reward_map, mdp.transition_table

    # for i in range(k):
    while True:
        delta = 0

        for s in states:
            old_util = utility[s]
            utility[s] = R[s[0]][s[1]] + gamma * sum([p * utility[s1] for (p, s1, r) in T[s][action_codes[pi[s]]]])

            delta = max(delta, abs(old_util - utility[s]))

        if delta < epsilon:
            break

    return utility


def policy_iteration(mdp: AngryBirds, gamma=0.9) -> Dict[Tuple[int, int], Tuple[int, int]]:
    """Solve an MDP by policy iteration"""
    states = get_states_from_transitions(mdp.transition_table)
    # utility = {s: 0 for s in states}
    utility = {s: mdp.reward_map[s[0]][s[1]] for s in states}
    pi = {s: (0, 1) for s in states}  # Prefer moving right initially.

    while True:
        utility = policy_evaluation(pi, utility, mdp, states, gamma)
        unchanged = True

        for s in states:
            best_action = actions[0]
            value = - float('inf')

            for a in actions:
                e = expected_utility(a, s, utility, mdp)
                if e > value:
                    value = e
                    best_action = a

            # action = argmax(actions, key=lambda a: expected_utility(a, s, utility, mdp))
            if best_action != pi[s]:
                pi[s] = best_action
                unchanged = False

        if unchanged:
            return pi


def value_iteration(mdp: AngryBirds, gamma=0.9, epsilon=1e-4) -> Tuple[Dict[Tuple[int, int], Tuple[int, int]], List[int], List[float]]:
    """Perform Value Iteration to determine the optimal policy."""
    states = get_states_from_transitions(mdp.transition_table)
    utility = {s: mdp.reward_map[s[0]][s[1]] for s in states}
    policy = {s: (0, 1) for s in states}

    value_differences = []  # To track Value Difference at each iteration

    while True:
        delta = 0  # Track maximum utility change per iteration.
        new_utility = utility.copy()
        value_difference = 0  # For calculating the sum of absolute differences

        for s in states:
            # Compute the maximum expected utility for each possible action.
            action_values = {}
            for a in actions:
                sum_ = 0
                for (p, s1, r) in mdp.transition_table[s][action_codes[a]]:
                    sum_ += p * (r + gamma * utility[s1])
                action_values[a] = sum_

            # Update the utility with the max action value.
            new_utility[s] = max(action_values.values())
            # Track the best action for the policy.
            policy[s] = max(action_values, key=action_values.get)
            # Track the total absolute difference for Value Difference.
            value_difference += abs(new_utility[s] - utility[s])
            # Calculate the maximum change in utility.
            delta = max(delta, abs(new_utility[s] - utility[s]))

        # Save Value Difference for this iteration
        value_differences.append(value_difference)

        # Update utilities for the next iteration.
        utility = new_utility

        # Stop when the utilities converge (change < epsilon).
        if delta < epsilon:
            break

    v_star = [utility[(i, j)] for i in range(8) for j in range(8)]
    return policy, v_star, value_differences
