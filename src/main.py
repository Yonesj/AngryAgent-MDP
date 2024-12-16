import os
import pygame

from environment import PygameInit, AngryBirds
from policy import value_iteration
from const import action_codes, FPS
from logger import MDPLogger
from util import save_value_difference_plot


def main():
    env = AngryBirds()
    screen, clock = PygameInit.initialization()
    state = env.reset()

    policy, v_star, value_differences = value_iteration(env, .6)
    save_value_difference_plot(value_differences)
    # print_rewards(env.reward_map)
    # print_policy(policy)
    total_score = 0

    for i in range(5):
        running = True
        reward_episode = 0
        steps = 0

        logger = MDPLogger(fr'..\logs\log{i + 1}.txt')
        logger.log(env.reward_map, policy, v_star)

        while running:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            env.render(screen)

            # extract action from policy
            action = action_codes[policy[state]]
            state, probability, reward, done = env.step(action)
            reward_episode += reward
            steps += 1

            if reward == 250 or steps >= 40:
                env.update_transition_table(state)
                policy, v_star, _ = value_iteration(env, .6)
                logger.log(env.reward_map, policy, v_star)
                # print_rewards(env.reward_map)
                # print_policy(policy)
                steps = 0

            if done:
                print(f"Episode finished with reward: {reward_episode}")
                state = env.reset()
                break

            pygame.display.flip()
            clock.tick(FPS)

        total_score += reward_episode

    print(f"mean: {total_score / 5}")
    pygame.quit()


if __name__ == "__main__":
    os.makedirs("../logs", exist_ok=True)
    os.makedirs("../plot", exist_ok=True)

    main()
