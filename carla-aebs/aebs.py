# Python module to demonstrate AEBS
import click
import numpy as np
import os

from models.rl_agent.ddpg_agent import ddpgAgent
from models.rl_agent.input_preprocessor import InputPreprocessor
from utils.visualize import plot_metrics
from world import World


@click.command()
@click.option(
    '--mode',
    type=click.Choice(['out', 'in'], case_sensitive=False), default='in',
    help='Mode to run the simulation in (Out of distribution / Normal)'
)
@click.option('--gui', is_flag=True, default=False, help='Run simulation with GUI')
@click.option('--testing', is_flag=True, default=False, help='Run simulation in testing mode')
@click.option('--save-path', help='Path to save checkpoints to. Only used during training mode')
@click.option('--agent-chkpt-path', help='Path to load checkpoint for the RL agent')
@click.option('--perception-chkpt-path', help='Path to load checkpoint for the Perception LEC')
@click.option('--vae-chkpt-path', help='Path to load checkpoint for the Perception LEC')
@click.option('--num-episodes', type=int, default=1, help='Number of episodes to run tests for.')
@click.option('--generate-plots', is_flag=True, default=False, help='Generate plots after completing the simulation')
def aebs(
    gui=False, testing=False, num_episodes=1,
    save_path=os.getcwd(), mode='in', agent_chkpt_path=None,
    perception_chkpt_path=None, vae_chkpt_path=None, generate_plots=False
):
    """Command to run simulation in train/test mode.
    For collecting data please refer to the collect command.

    Sample Usage: python aebs.py --save-path /home/lexent/carla_simulation/rl_agent/ \
                                --num-episodes 1 \
                                --agent-chkpt-path /home/lexent/carla_simulation/rl_agent/ \
                                --perception-chkpt-path /home/lexent/carla_simulation/perception_chkpt/chkpt_8.pt \
                                --vae-chkpt-path /home/lexent/carla_simulation/vae_chkpt/chkpt_92.pt \
                                --gui --testing --generate-plots

    Args:
        gui (bool, optional): [Run simulation with GUI]. Defaults to False.\n
        testing (bool, optional): [Run simulation in testing mode]. Defaults to False.\n
        num_episodes (int, optional): [Number of episodes to run tests for.]. Defaults to 1.\n
        save_path ([type], optional): [Path to save checkpoints to. Only used during training mode]. Defaults to os.getcwd().\n
        mode (str, optional): [Mode to run the simulation in (Out of distribution / Normal)]. Defaults to 'in'.\n
        agent_chkpt_path ([type], optional): [Path to load checkpoint for the RL agent]. Defaults to None.\n
    """
    agent = ddpgAgent(testing=testing, load_path=agent_chkpt_path)
    world = World(
        gui=gui, collect=False, testing=testing,
        perception_chkpt=perception_chkpt_path, vae_chkpt=vae_chkpt_path
    )
    input_preprocessor = InputPreprocessor()
    if mode == 'in':
        ppt_lower_limit = 0
        ppt_upper_limit = 20
    elif mode == 'out':
        ppt_lower_limit = 60
        ppt_upper_limit = 100

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
        actions = []
        while True:
            a = agent.getAction(s, epsilon)
            actions.append(a[0][0])
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
                print(f"Episode {episode + 1} is done, the reward is {reward}")
                break

    # Generate plots after the simulation ends for the last episode
    if generate_plots:
        comp_distances = np.array(world.computed_distances)
        gt_distances = np.array(world.gt_distances)
        p_values = np.array(world.p_values)
        actions = np.array(actions)
        plot_metrics(comp_distances, gt_distances, actions, p_values)


if __name__ == '__main__':
    aebs()
