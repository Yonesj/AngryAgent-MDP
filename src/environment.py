import random
import copy
from typing import List, Tuple

import numpy as np
import pygame

from const import *
from util import PriorityQueue, manhattan_distance


class PygameInit:
    """
    Initializes Pygame with a specified grid and tile size.

    Methods:
        initialization():
            Sets up the Pygame window and returns the screen and clock.
    """

    @classmethod
    def initialization(cls) -> Tuple[pygame.Surface, pygame.time.Clock]:
        """
        Initializes Pygame screen and clock.

        Returns:
            Tuple[pygame.Surface, pygame.time.Clock]: Pygame screen and clock objects.
        """
        grid_size = 8
        tile_size = 100

        pygame.init()
        screen = pygame.display.set_mode((grid_size * tile_size, grid_size * tile_size))
        pygame.display.set_caption("MDP Angry Birds")
        clock = pygame.time.Clock()

        return screen, clock


class AngryBirds:
    """
    Represents the Angry Birds environment for a Markov Decision Process (MDP).

    Attributes:
        actions (dict): Mapping of action indices to grid movements.
        neighbors (dict): Neighboring actions for each action.

    Methods:
        reset():
            Resets the environment to its initial state.
        step(action):
            Executes an action and returns the next state, probability, reward, and termination status.
        render(screen):
            Renders the environment grid and agent on the given Pygame screen.
        reward_function():
            Computes the reward map for the environment.
    """

    actions = {
        0: (-1, 0),  # Up
        1: (1, 0),  # Down
        2: (0, -1),  # Left
        3: (0, 1)  # Right
    }

    neighbors = {
        0: [2, 3],  # Up -> Left and Right
        1: [2, 3],  # Down -> Left and Right
        2: [0, 1],  # Left -> Up and Down
        3: [0, 1]  # Right -> Up and Down
    }

    def __init__(self):
        """
        Initializes the Angry Birds environment with default parameters and assets.
        """
        self.__grid_size = 8
        self.__tile_size = 100
        self.__num_pigs = PIGS
        self.__num_queens = QUEENS
        self.__num_rocks = ROCKS

        self.__probability_dict = self.__generate_probability_dict()
        self.__base_grid = self.__generate_grid()
        self.__agent_pos = (0, 0)

        self.grid = copy.deepcopy(self.__base_grid)
        self.reward = 0
        self.done = False
        self.reward_map = self.__reward_function(self.__agent_pos)
        self.transition_table = self.__calculate_transition_model(self.__grid_size, self.__probability_dict,
                                                                  self.reward_map)

        self.__load_assets()

    def __load_assets(self):
        """
        Loads and processes image assets for the game.
        """
        self.__agent_image, _ = self.__load_and_scale_image("../assets/angry-birds.png")
        self.__pig_image, self.__pig_with_background = self.__load_and_scale_image('../assets/pigs.png', True)
        self.__egg_image, self.__egg_with_background = self.__load_and_scale_image('../assets/eggs.png', True)
        self.__queen_image, self.__queen_with_background = self.__load_and_scale_image('../assets/queen.png', True)
        self.__rock_image, self.__rock_with_background = self.__load_and_scale_image('../assets/rocks.png', True)

    def __load_and_scale_image(self, path: str, with_background: bool = False) -> Tuple[pygame.Surface, pygame.Surface]:
        """
        Loads and scales an image, optionally applying a background.

        Args:
            path (str): Path to the image file.
            with_background (bool): Whether to apply a background.

        Returns:
            pygame.Surface: The processed image surface.
        """
        image = pygame.image.load(path)
        image = pygame.transform.scale(image, (self.__tile_size, self.__tile_size))
        surface = None

        if with_background:
            surface = pygame.Surface((self.__tile_size, self.__tile_size))
            surface.fill((135, 206, 235))
            surface.blit(image, (0, 0))

        return image, surface

    def __generate_grid(self) -> List[List[str]]:
        """
        Generates the initial grid for the environment.

        Returns:
            List[List[str]]: A 2D list representing the grid.
        """
        while True:
            filled_spaces = [(0, 0), (self.__grid_size - 1, self.__grid_size - 1)]
            grid = [['T' for _ in range(self.__grid_size)] for _ in range(self.__grid_size)]

            self.__populate_grid(grid, 'P', self.__num_pigs, filled_spaces)
            self.__populate_grid(grid, 'Q', self.__num_queens, filled_spaces)
            self.__populate_grid(grid, 'R', self.__num_rocks, filled_spaces)

            grid[self.__grid_size - 1][self.__grid_size - 1] = 'G'

            if self.__is_path_exists(grid=grid, start=(0, 0), goal=(7, 7)):
                break

        return grid

    def __populate_grid(self, grid: List[List[str]], symbol: str, count: int, filled_spaces: List[Tuple[int, int]]) -> None:
        """
        Populates the grid with a specific symbol at random locations.

        Args:
            grid (List[List[str]]): The grid to populate.
            symbol (str): The symbol to place in the grid.
            count (int): Number of symbols to place.
            filled_spaces (List[Tuple[int, int]]): List of already filled spaces.
        """
        for _ in range(count):
            while True:
                r, c = random.randint(0, self.__grid_size - 1), random.randint(0, self.__grid_size - 1)
                if (r, c) not in filled_spaces:
                    grid[r][c] = symbol
                    filled_spaces.append((r, c))
                    break

    def reset(self) -> Tuple[int, int]:
        """
        Resets the environment to its initial state.

        Returns:
            Tuple[int, int]: The agent's initial position.
        """
        self.grid = copy.deepcopy(self.__base_grid)
        self.__agent_pos = (0, 0)
        self.reward = 0
        self.done = False
        return self.__agent_pos

    def step(self, action: int) -> Tuple[Tuple[int, int], float, int, bool]:
        """
        Executes the given action and returns the next state, probability, reward, and termination status.

        Args:
            action (int): The action to execute (0: Up, 1: Down, 2: Left, 3: Right).

        Returns:
            Tuple[Tuple[int, int], float, int, bool]:
                - Next state (row, col)
                - Probability of the chosen action
                - Reward received
                - Termination status
        """
        prob_dist = self.__get_action_probability_distribution(action)
        chosen_action = np.random.choice([0, 1, 2, 3], p=prob_dist)

        dx, dy = self.actions[chosen_action]
        new_row, new_col = self.__agent_pos[0] + dx, self.__agent_pos[1] + dy

        if (0 <= new_row < self.__grid_size and 0 <= new_col < self.__grid_size and
                self.grid[new_row][new_col] != 'R'):
            self.__agent_pos = (new_row, new_col)

        current_tile = self.grid[self.__agent_pos[0]][self.__agent_pos[1]]
        reward = self.__get_reward_for_tile(current_tile)
        # update the tile
        self.grid[self.__agent_pos[0]][self.__agent_pos[1]] = 'T'

        self.reward = reward
        return self.__agent_pos, prob_dist[chosen_action], self.reward, self.done

    def __get_action_probability_distribution(self, action: int) -> List[float]:
        """
        Gets the probability distribution of possible actions based on the intended action.

        Args:
            action (int): The intended action index (0: Up, 1: Down, 2: Left, 3: Right).

        Returns:
            List[float]: A list representing the probability distribution of actions.
        """
        prob_dist = [0] * 4
        prob_dist[action] = self.__probability_dict[self.__agent_pos][action]['intended']

        for neighbor_action in self.neighbors[action]:
            prob_dist[neighbor_action] = self.__probability_dict[self.__agent_pos][action]['neighbor']

        return prob_dist

    def __get_reward_for_tile(self, tile: str) -> int:
        """
        Gets the reward associated with the given tile type.

        Args:
            tile (str): The type of tile ('P', 'Q', 'G', 'T', or 'R').

        Returns:
            int: The reward value for the tile.
        """
        if tile == 'Q':
            return QUEEN_REWARD
        elif tile == 'P':
            return GOOD_PIG_REWARD
        elif tile == 'G':
            self.done = True
            return GOAL_REWARD

        return DEFAULT_REWARD

    def render(self, screen: pygame.Surface) -> None:
        """
        Renders the environment grid and agent on the given Pygame screen.

        Args:
            screen (pygame.Surface): The Pygame screen to render on.
        """
        for r in range(self.__grid_size):
            for c in range(self.__grid_size):
                color = COLORS[self.grid[r][c]]
                pygame.draw.rect(screen, color,
                                 (c * self.__tile_size, r * self.__tile_size, self.__tile_size, self.__tile_size))

                self.__render_tile(screen, r, c)

        self.__draw_grid_lines(screen)
        screen.blit(self.__agent_image,
                    (self.__agent_pos[1] * self.__tile_size, self.__agent_pos[0] * self.__tile_size))

    def __render_tile(self, screen: pygame.Surface, row: int, col: int) -> None:
        """
        Renders specific tiles based on their type.

        Args:
            screen (pygame.Surface): The Pygame screen.
            row (int): Row index.
            col (int): Column index.
        """
        tile = self.grid[row][col]
        tile_mapping = {
            'P': self.__pig_with_background,
            'G': self.__egg_with_background,
            'Q': self.__queen_with_background,
            'R': self.__rock_with_background
        }

        if tile in tile_mapping:
            screen.blit(tile_mapping[tile], (col * self.__tile_size, row * self.__tile_size))

    def __draw_grid_lines(self, screen: pygame.Surface) -> None:
        """
        Draws grid lines on the Pygame screen.

        Args:
            screen (pygame.Surface): The Pygame screen.
        """
        for r in range(self.__grid_size + 1):
            pygame.draw.line(screen, (0, 0, 0), (0, r * self.__tile_size),
                             (self.__grid_size * self.__tile_size, r * self.__tile_size), 2)
        for c in range(self.__grid_size + 1):
            pygame.draw.line(screen, (0, 0, 0), (c * self.__tile_size, 0),
                             (c * self.__tile_size, self.__grid_size * self.__tile_size), 2)

    def __reward_function(self, current_state: Tuple[int, int]) -> List[List[int]]:
        """
        Computes the reward map for the environment.

        Args:
            current_state (Tuple[int, int])
        Returns:
            List[List[int]]: The reward map for each grid cell.
        """
        eggs_pos = (7, 7)
        goal_pos = None
        min_distance = 14

        for i in range(8):
            for j in range(8):
                if self.grid[i][j] == 'P':  # Pig
                    d = manhattan_distance((i, j), current_state)
                    if d < min_distance:
                        goal_pos = (i, j)
                        min_distance = d

        if goal_pos is None:
            goal_pos = eggs_pos

        reward_table = [[0 for _ in range(self.__grid_size)] for _ in range(self.__grid_size)]

        # Populate static rewards based on the grid
        for i in range(self.__grid_size):
            for j in range(self.__grid_size):
                if (i, j) == goal_pos:
                    reward_table[i][j] = 1000
                elif self.grid[i][j] == 'P':  # Pig
                    reward_table[i][j] = 250
                elif self.grid[i][j] == 'Q':  # Queen Pig
                    reward_table[i][j] = -400
                elif self.grid[i][j] == 'G':  # Goal
                    reward_table[i][j] = 400
                elif self.grid[i][j] == 'R':  # W
                    reward_table[i][j] = None  # Impassable

        # A* Search initialization
        initial_pos = (0, 0)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        frontiers = PriorityQueue()
        reached_states = {initial_pos: 0}
        frontiers.push((initial_pos, []), 0)

        while not frontiers.is_empty():
            current_pos, path = frontiers.pop()
            x, y = current_pos

            if (x, y) == goal_pos:
                result = [[0 for _ in range(self.__grid_size)] for _ in range(self.__grid_size)]

                for i in range(self.__grid_size):
                    for j in range(self.__grid_size):
                        if (i, j) in reached_states:
                            result[i][j] = max(reached_states[(i, j)], (14 - manhattan_distance((i, j), goal_pos)) * 10)
                        elif reward_table[i][j] is not None:
                            result[i][j] = reward_table[i][j] + (14 - manhattan_distance((i, j), goal_pos)) * 10

                for x, y in path:
                    result[x][y] += 100
                return result

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                next_pos = (nx, ny)

                if (
                        0 <= nx < self.__grid_size and
                        0 <= ny < self.__grid_size and
                        reward_table[nx][ny] is not None
                ):
                    new_reward = reached_states[(x, y)] + reward_table[nx][ny]

                    if next_pos not in reached_states or new_reward > reached_states[next_pos]:
                        if self.grid[nx][ny] == 'P':
                            reward_table[nx][ny] = 0
                        reached_states[next_pos] = new_reward
                        # frontiers.push((nx, ny), -new_reward + (manhattan_distance((nx, ny), goal_pos) * 10))
                        frontiers.push((next_pos, path + [next_pos]), -new_reward + (manhattan_distance(next_pos, goal_pos)) * 10)

        raise Exception("Failed to calculate rewards for actions")

    @classmethod
    def __calculate_transition_model(cls, grid_size: int, actions_prob: dict, reward_map: List[List[int]]) -> dict:
        """
        Calculates the transition model for the environment.

        Args:
            grid_size (int): The size of the grid.
            actions_prob (dict): Action probabilities for each state.
            reward_map (List[List[int]]): Reward values for the grid cells.

        Returns:
            dict: A transition table containing probabilities, next states, and rewards for each action.
        """
        transition_table = {}

        for row in range(grid_size):
            for col in range(grid_size):
                state = (row, col)
                transition_table[state] = {}

                for action in range(4):
                    transition_table[state][action] = []

                    intended_move = actions[action]
                    next_state = (row + intended_move[0], col + intended_move[1])

                    if 0 <= next_state[0] < grid_size and 0 <= next_state[1] < grid_size:
                        reward = reward_map[next_state[0]][next_state[1]]
                        intended_probability = actions_prob[(next_state[0], next_state[1])][action]['intended']
                        transition_table[state][action].append((intended_probability, next_state, reward))
                    else:
                        intended_probability = actions_prob[state][action]['intended']
                        transition_table[state][action].append(
                            (intended_probability, state, reward_map[row][col]))

                    for neighbor_action in cls.neighbors[action]:
                        neighbor_move = actions[neighbor_action]
                        next_state = (row + neighbor_move[0], col + neighbor_move[1])

                        if 0 <= next_state[0] < grid_size and 0 <= next_state[1] < grid_size:
                            reward = reward_map[next_state[0]][next_state[1]]
                            neighbor_probability = actions_prob[(next_state[0], next_state[1])][action]['neighbor']
                            transition_table[state][action].append((neighbor_probability, next_state, reward))
                        else:
                            neighbor_probability = actions_prob[state][action]['neighbor']
                            transition_table[state][action].append(
                                (neighbor_probability, state, reward_map[row][col]))

        return transition_table

    def update_transition_table(self, current_state: Tuple[int, int]) -> None:
        self.reward_map = self.__reward_function(current_state)

        for i in range(8):
            for j in range(8):
                for a in range(4):
                    possible_outcome = self.transition_table[(i, j)][a]
                    updated_values = []

                    for (probability, next_state, _) in possible_outcome:
                        updated_values.append((probability, next_state, self.reward_map[i][j]))

                    self.transition_table[(i, j)][a] = updated_values

    @classmethod
    def __is_path_exists(cls, grid: List[List[str]], start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
        """
        Checks if a valid path exists from the start position to the goal position.

        Args:
            grid (List[List[str]]): The grid representation of the environment.
            start (Tuple[int, int]): The starting position.
            goal (Tuple[int, int]): The goal position.

        Returns:
            bool: True if a valid path exists, False otherwise.
        """
        grid_size = len(grid)
        visited = set()

        def dfs(x, y):
            if (x, y) == goal:
                return True
            visited.add((x, y))

            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (0 <= nx < grid_size and 0 <= ny < grid_size and
                        (nx, ny) not in visited and grid[nx][ny] != 'R'):
                    if dfs(nx, ny):
                        return True
            return False

        return dfs(start[0], start[1])

    def __generate_probability_dict(self):
        probability_dict = {}

        for row in range(self.__grid_size):
            for col in range(self.__grid_size):
                state = (row, col)
                probability_dict[state] = {}

                for action in range(4):
                    intended_prob = random.uniform(0.60, 0.80)
                    remaining_prob = 1 - intended_prob
                    neighbor_prob = remaining_prob / 2

                    probability_dict[state][action] = {
                        'intended': intended_prob,
                        'neighbor': neighbor_prob
                    }
        return probability_dict
