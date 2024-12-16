# Utility functions and datastructures for the Angry Birds MDP.
import heapq
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt


class PriorityQueue:
    """
      Implements a priority queue data structure. Each inserted item
      has a priority associated with it and the client is usually interested
      in quick retrieval of the lowest-priority item in the queue. This
      data structure allows O(1) access to the lowest-priority item.
    """

    def __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def is_empty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)


def manhattan_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """
    Calculate the Manhattan distance between two points.

    Args:
        a (Tuple[int, int]): The first point (row, col).
        b (Tuple[int, int]): The second point (row, col).

    Returns:
        int: The Manhattan distance between points `a` and `b`.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def save_value_difference_plot(value_differences: List[float], file_path: str = "../plot/value_difference_plot.png") -> None:
    """
    Save the Value Difference plot as an image file.
    Args:
        value_differences (List[float]): A list of value differences for each iteration.
        file_path (str): The file path where the plot will be saved (default: "../plot/value_difference_plot.png").
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(value_differences) + 1), value_differences, marker='o')
    plt.title("Value Difference over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Value Difference")
    plt.grid(True)
    plt.savefig(file_path, format='png')
    plt.close()  # Close the plot to free memory


def print_rewards(reward_table: List[List[int]]) -> None:
    """
    Print the reward table in a readable grid format.

    Args:
        reward_table (List[List[int]]): A 2D list representing the reward table.
    """
    print(' ' * 4, end='')
    for i in range(8):
        print(f"[{str(i)}]".rjust(6), end='\t')

    print()

    for i in range(8):
        print(f"[{i}]", end='\t')
        for j in range(8):
            print(f"[{str(reward_table[i][j]).rjust(4)}]", end='\t')
        print()
    print()


def print_policy(policy: Dict[Tuple[int, int], Tuple[int, int]]) -> None:
    """
    Print the policy table in a readable format with directions for each grid cell.

    Args:
        policy (Dict[Tuple[int, int], Tuple[int, int]]):
            A dictionary mapping each grid cell (row, col) to an action (delta_row, delta_col).
    """
    for i in range(8):
        for j in range(8):
            if policy[(i, j)] == (-1, 0):
                print(f"{(i, j)}: " + "UP".ljust(5), end="\t")
            elif policy[(i, j)] == (1, 0):
                print(f"{(i, j)}: " + "DOWN".ljust(5), end="\t")
            elif policy[(i, j)] == (0, -1):
                print(f"{(i, j)}: " + "LEFT".ljust(5), end="\t")
            else:
                print(f"{(i, j)}: RIGHT", end="\t")
        print()
    print()
