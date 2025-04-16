import pathlib
import argparse
import time
from ruamel.yaml import YAML

import torch

from stable_baselines3.common.logger import configure

from lapal.utils import utils, replay_buffer
from lapal.models.sac_ae import SacAeAgent

def evaluate(env, agent, num_episodes, L, step):
    for i in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = agent.sample_action(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward   

        L.record('eval/episode_reward', episode_reward)
    L.dump(step=step)

def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--config', help='train config file path')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    yaml = YAML(typ='safe')
    params = yaml.load(open(args.config))

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################
    if params['expert_folder'] is not None:
        demo_dir = (pathlib.Path(params['expert_folder']) / 
            params['env_name'] / params['robots'] / params['controller_type']).resolve()

    data_path = pathlib.Path(__file__).parent.parent.parent / 'data' / time.strftime("%m.%d.%Y")
    logdir = '_'.join([
        time.strftime("%H-%M-%S"),
        params['env_name'],
        params['controller_type'],
        params['suffix']
    ])
    logdir = data_path / logdir
    params['logdir'] = str(logdir)
    print(params)

    # dump params
    logdir.mkdir(parents=True, exist_ok=True)
    import yaml
    with open(logdir / 'params.yml', 'w') as fp:
        yaml.safe_dump(params, fp, sort_keys=False)

    model_dir = logdir / 'models'
    pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)

    ##################################
    ### SETUP ENV, AGENT
    ##################################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = utils.make(
        params['env_name'], 
        robots=params['robots'],
        controller_type=params['controller_type'],
        obs_keys=params['obs_keys'], 
        seed=params['seed'],
        **params['env_kwargs'],
    )
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    print(f"Environment observation space shape {obs_shape}")
    print(f"Environment action space shape {act_shape}")

    eval_env = utils.make(
        params['env_name'], 
        robots=params['robots'],
        controller_type=params['controller_type'],
        obs_keys=params['obs_keys'], 
        seed=params['seed']+100,
        **params['env_kwargs'],
    )
    logger = configure(params['logdir'], ["stdout", "tensorboard"])

    obs_encoder_params = utils.AttrDict(
        type=params['ob_ae']['type'],
        input_dim=obs_shape[0],
        feature_dim=params['ob_ae']['feature_dim'],
        n_layers=params['ob_ae']['n_layers'],
        hidden_dim=params['ob_ae']['hidden_dim'],
        activation=params['ob_ae']['activation'],
        output_activation=params['ob_ae']['output_activation'],
        lr=params['ob_ae']['lr'],
        update_freq=params['ob_ae']['update_freq'],
    )
    act_encoder_decoder_params = utils.AttrDict(
        type=params['ac_ae']['type'],
        input_dim=act_shape[0],
        feature_dim=params['ac_ae']['feature_dim'],
        cond_feature_dim=obs_shape[0],
        n_layers=params['ac_ae']['n_layers'],
        hidden_dim=params['ac_ae']['hidden_dim'],
        activation=params['ac_ae']['activation'],
        output_activation=params['ac_ae']['output_activation'],
        lr=params['ac_ae']['lr'],
        update_freq=params['ac_ae']['update_freq'],
        kl_coef=params['ac_ae']['kl_coef'],
    )
    actor_critic_params = utils.AttrDict(
        lr=params['actor_critic']['lr'], 
        n_layers=params['actor_critic']['n_layers'], 
        hidden_dim=params['actor_critic']['hidden_dim'],
        update_freq=params['actor_critic']['update_freq'],
        tau=0.005,
    )
    disc_params = utils.AttrDict(
        reward_type=params['discriminator']['reward_type'], 
        n_layers=params['discriminator']['n_layers'],
        hidden_dim=params['discriminator']['hidden_dim'],
        activation=params['discriminator']['activation'],
        spectral_norm=params['discriminator']['spectral_norm'],
        lr=params['discriminator']['lr'],
        update_freq=params['discriminator']['update_freq'],
    )

    agent = SacAeAgent(
        obs_shape,
        act_shape,
        device=device,
        obs_encoder_params=obs_encoder_params,
        act_encoder_decoder_params=act_encoder_decoder_params,
        actor_critic_params=actor_critic_params,
        disc_params=disc_params,
    )

    agent_replay_buffer = replay_buffer.ReplayBuffer(
        obs_shape=obs_shape,
        action_shape=act_shape,
        capacity=1000000,
        batch_size=params['batch_size'],
        device=device
    )

    demo_replay_buffer = replay_buffer.ReplayBuffer(
        obs_shape=obs_shape,
        action_shape=act_shape,
        capacity=64000,
        batch_size=params['batch_size'],
        device=device
    )

    if params['expert_folder'] is not None:
        demo_episodes = utils.load_episodes(demo_dir, params['obs_keys'])
        agent_replay_buffer.add_rollouts(demo_episodes)
        demo_replay_buffer.add_rollouts(demo_episodes)

    if params['save_buffer']:
        replay_buffer_dir = logdir / 'replay_buffer'
        pathlib.Path(replay_buffer_dir).mkdir(parents=True, exist_ok=True)

    print(f'Latent observation dim: {agent.obs_encoder.feature_dim}')
    print(f'Latent action dim: {agent.act_encoder.feature_dim}')
        
    ##################################
    ### RUN TRAINING
    ##################################

    episode, episode_reward, done = 0, 0, True
    start_time = time.time()
    for step in range(params['total_timesteps']):
        if done:
            if step > 0:
                logger.record('train/duration', time.time() - start_time)
                start_time = time.time()
                logger.dump(step=step) 
            if step % params['evaluation']['interval'] == 0:
                print(f"Evaluating at step {step}")
                logger.record('eval/episode', episode)
                evaluate(eval_env, agent, 1, logger, step) 
            if step % params['evaluation']['save_interval'] == 0:
                print(f"Saving model at step {step}")
                step_dir = model_dir / f"step_{step:07d}"
                pathlib.Path(step_dir).mkdir(parents=True, exist_ok=True)
                agent.save(step_dir)
            if params['save_buffer']:
                agent_replay_buffer.save(replay_buffer_dir)

            logger.record('train/episode_reward', episode_reward)            

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1   
            logger.record('train/episode', episode)

        # run training update
        if step >= params['pretrain_steps']:
            agent.update(agent_replay_buffer, demo_replay_buffer, logger, step) 

        # sample action for data collection        
        if step < params['pretrain_steps']:
            action = env.action_space.sample()
        else:
            action = agent.sample_action(obs)
        next_obs, reward, done, _ = env.step(action)

        # allow infinit bootstrap
        done_bool = 0 
        episode_reward += reward

        agent_replay_buffer.add(obs, action, reward, next_obs, done_bool) 

        obs = next_obs
        episode_step += 1

if __name__ == '__main__':
    main()