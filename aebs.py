# Python module to demonstrate AEBS
import click
import numpy as np
import os

from tqdm import tqdm
from world import World
from rl_agent.ddpg_agent import ddpgAgent
from rl_agent.input_preprocessor import InputPreprocessor


@click.command()
@click.option('--mode', type=click.Choice(['out', 'in'], case_sensitive=False), default='in')
@click.option('--gui', is_flag=True, default=False)
@click.option('--testing', is_flag=True, default=False)
@click.option('--save-path')
@click.option('--load-path')
@click.option('--num-episodes',type=int, default=1)
def aebs(gui=False, testing=False, num_episodes=1, save_path=os.getcwd(), mode='in', load_path=None):
    world = World(gui=gui, collect=False)
    agent = ddpgAgent(testing=testing, load_path=load_path)
    input_preprocessor = InputPreprocessor()
    if mode == 'in':
        ppt_lower_limit=0
        ppt_upper_limit=20
    elif mode == 'out':
        ppt_lower_limit=60
        ppt_upper_limit=100

    best_reward = -1000  # Any large negative value will do

    for episode in range(num_episodes):
        print(f'Running episode:{episode + 1}')
        # Sample random distance and velocity values
        initial_distance = np.random.normal(100, 1)
        initial_velocity = np.random.uniform(25, 28)

        # Sample a random precipitation parameter
        precipitation = np.random.uniform(ppt_lower_limit, ppt_upper_limit)
        print(f'Precipitation: {precipitation}')

        # Initialize the world with the sampled params
        dist, vel, status = world.init(initial_velocity, initial_distance, precipitation)
        if status == 'FAILED':
            print(f'Reset failed. Stopping episode {episode + 1} and continuing!')
            continue
        
        # Setup the starting state based on the state returned by resetting the world
        s = (dist, vel)
        s = input_preprocessor(s)
        epsilon = 1.0 - (episode+1) / num_episodes
        time_step = 0
        while True:
            a = agent.getAction(s, epsilon)
            dist, vel, reward, episode_status = world.step(brake=a[0][0])
            s_ = (dist, vel)
            s_ = input_preprocessor(s_)
            if testing is False:
                # Train the agent if testing is disabled
                agent.storeTrajectory(s, a, reward, s_, episode_status)
                agent.learn()
            s = s_
            if episode_status == "DONE":
                if reward > best_reward and testing is False:
                    best_save_path = os.path.join(save_path, 'best')
                    os.makedirs(best_save_path, exist_ok=True)
                    agent.save_model(best_save_path)
                    best_reward = reward
                if testing is False:
                    if np.mod(episode, 10) == 0:
                        agent.save_model(save_path)
                print(f"Episode {episode} is done, the reward is {reward}")
                break


if __name__ == '__main__':
    aebs()
