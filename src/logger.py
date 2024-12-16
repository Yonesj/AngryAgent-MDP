from typing import List, Dict, Tuple


class MDPLogger:
    def __init__(self, file_name: str):
        self.__file_name = file_name
        self.__iteration = 1  # Start iteration from 1

        # Clear the contents of the file or create a new one
        with open(self.__file_name, 'w') as file:
            file.write("MDP Logging Start\n")
            file.write("=" * 40 + "\n\n")

    def log(self, reward_table: List[List[int]],
            policy: Dict[Tuple[int, int], Tuple[int, int]],
            value_star: List[int]) -> None:
        """Logs reward table, policy, and value_star to a file."""
        with open(self.__file_name, 'a') as file:
            file.write(f"Iteration {self.__iteration}\n")
            file.write("=" * 30 + "\n")

            self._log_rewards(file, reward_table)
            self._log_policy(file, policy)
            self._log_value_star(file, value_star)

            file.write("\n" + "=" * 30 + "\n\n")

        # Increment iteration count
        self.__iteration += 1

    @staticmethod
    def _log_rewards(file, reward_table: List[List[int]]) -> None:
        """Logs the reward table."""
        file.write("Reward Table:\n")
        file.write(' ' * 4 + ''.join(f"[{i}]".rjust(6) + '\t' for i in range(8)) + '\n')

        for i in range(8):
            file.write(f"[{i}]\t" + ''.join(f"[{str(reward_table[i][j]).rjust(4)}]\t" for j in range(8)) + '\n')
        file.write('\n')

    @staticmethod
    def _log_policy(file, policy: Dict[Tuple[int, int], Tuple[int, int]]) -> None:
        """Logs the policy."""
        file.write("Policy:\n")
        directions = {(-1, 0): "UP", (1, 0): "DOWN", (0, -1): "LEFT", (0, 1): "RIGHT"}

        for i in range(8):
            for j in range(8):
                action = policy.get((i, j), (0, 0))
                direction = directions.get(action, "STAY")
                file.write(f"{(i, j)}: {direction.ljust(6)}\t")
            file.write('\n')
        file.write('\n')

    @staticmethod
    def _log_value_star(file, value_star: List[int]) -> None:
        """Logs the value function."""
        file.write("Value Function (Value*):\n")
        for i in range(8):
            for j in range(8):
                idx = i * 8 + j
                value = value_star[idx] if idx < len(value_star) else 0
                file.write(f"{(i, j)}: {str(value).ljust(20)}\t")
            file.write('\n')
        file.write('\n')
