# Python module to collect data for the Perception net training
import click
import numpy as np
import os

from world import World


@click.option('--gui', is_flag=True, default=False, help='Enable GUI when collecting data')
@click.option('--collect-path', default=os.getcwd(), help='Directory in which the collected data will be stored')
@click.option('--num-episodes', type=int, default=1, help='Number of episodes over which to collect the data')
@click.option('--mode', type=click.Choice(['out', 'in'], case_sensitive=False), default='in', help='Mode in which to collect the data')
@click.command()
def collect(collect_path=os.getcwd(), gui=False, num_episodes=1, mode='in'):
    """Generates data required for training the Perception LEC
    For training the RL agent refer to the aebs command.

    Args:
        gui (bool, optional): [Enable GUI when collecting data]. Defaults to True.\n
        collect_path (str, optional): [Directory in which the collected data will be stored]. Defaults to os.cwdir().\n
        num_episodes (int, optional): [Number of episodes over which to collect the data]. Defaults to 1.\n
        mode (str, optional): [Mode in which to collect the data]. Defaults to 'in'.
    """
    # Create the world
    world = World(gui=gui, collect=collect, collect_path=collect_path)
    if mode == 'in':
        ppt_lower_limit = 0
        ppt_upper_limit = 20
    elif mode == 'out':
        ppt_lower_limit = 60
        ppt_upper_limit = 100

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
        while True:
            episode_status = world.step()
            if episode_status == "DONE":
                break


if __name__ == '__main__':
    collect()
