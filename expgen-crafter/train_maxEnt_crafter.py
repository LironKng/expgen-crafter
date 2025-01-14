import pathlib
import json
import wandb
from PPO_maxEnt_LEEP import algo, utils
from PPO_maxEnt_LEEP.arguments import get_args
from PPO_maxEnt_LEEP.model import Policy, ImpalaModel
from PPO_maxEnt_LEEP.storage import RolloutStorage
from PPO_maxEnt_LEEP.procgen_wrappers import *
from PPO_maxEnt_LEEP.logger import maxEnt_Logger
from PPO_maxEnt_LEEP.logger import Logger
import PPO_maxEnt_LEEP.hyperparams as hps
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PPO_maxEnt_LEEP.constant import TASKS
from evaluation import evaluate_procgen




from crafter.env import Env
from functools import partial
from PPO_maxEnt_LEEP.envs import VecPyTorch

EVAL_ENVS = ['train_eval', 'test_eval']

def main():
    args = get_args()
    import random
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    logdir_ = args.env_name + '_ppo' + '_seed_' + str(args.seed)
    logdir_ = logdir_ + '_maxEnt'
    logdir = os.path.join(os.path.expanduser(args.log_dir), logdir_)
    utils.cleanup_log_dir(logdir)

    print("logdir: " + logdir)
    print("printing args")
    argslog = pd.DataFrame(columns=['args', 'value'])
    for key in vars(args):
        log = [key] + [vars(args)[key]]
        argslog.loc[len(argslog)] = log
        print(key, ':', vars(args)[key])

    with open(logdir + '/args.csv', 'w') as f:
        argslog.to_csv(f, index=False)

    # progresslog = pd.DataFrame(columns=['timesteps', 'train intrinsic mean', 'train intrinsic min', 'train intrinsic max',
    #                                     'train extrinsic mean', 'train extrinsic min', 'train extrinsic max',
    #                                     'test intrinsic mean', 'test intrinsic min', 'test intrinsic max',
    #                                     'test extrinsic mean', 'test extrinsic min', 'test extrinsic max'])
    
    progresslog = pd.DataFrame(
    columns=['timesteps', 'train mean', 'train min', 'train max', 'test mean', 'test min', 'test max'])

    torch.set_num_threads(1)
    device = torch.device("cuda:{}".format(args.gpu_device) if args.cuda else "cpu")

    wandb.init(project=args.env_name + "_crafter", entity="liron-kng-technion-israel-institute-of-technology",
               config=args, name=logdir_, id=logdir_)

    print('making envs...')

    max_reward_seeds = {
        'train_eval': [],
        'test_eval': []
    }

    test_start_level = args.start_level + args.num_level + 1
    start_train_test = {
        'train_eval': args.start_level,
        'test_eval': test_start_level
    }

    eval_envs_dic = {}

    # Training envs
    seeds_train = np.random.randint(start_train_test['train_eval'], args.num_level, size=args.num_processes)
    env_fns = [partial(Env, seed=seed) for seed in seeds_train]
    envs = gym.vector.AsyncVectorEnv(env_fns)
    envs.observation_space = gym.spaces.Box(low=0, high=255, shape=(
        envs.observation_space.shape[1], envs.observation_space.shape[2], envs.observation_space.shape[3]),
                                            dtype=np.uint8)
    envs.action_space = envs.action_space[0]
    envs = TransposeFrame(envs)
    envs = VecPyTorch(envs, 'cuda')
    envs = ScaledFloatFrame(envs)
   

    # Test envs
    # Test environments are sampled from the full distribution of levels
    seeds_test = np.random.randint(start_train_test['test_eval'], start_train_test['test_eval'] + args.num_test_level, size=args.num_processes)
    env_fns = [partial(Env, seed=seed) for seed in seeds_test]
    test_env = gym.vector.AsyncVectorEnv(env_fns)
    test_env.observation_space = gym.spaces.Box(low=0, high=255, shape=(
        test_env.observation_space.shape[1], test_env.observation_space.shape[2], test_env.observation_space.shape[3]),
                                                dtype=np.uint8)
    test_env.action_space = test_env.action_space[0]
    test_env = TransposeFrame(test_env)
    test_env = VecPyTorch(test_env, 'cuda')
    test_env = ScaledFloatFrame(test_env)

    eval_envs_dic['train_eval'] = envs
    eval_envs_dic['test_eval'] = test_env

    # For success calculation
    total_successes = np.zeros((0, len(TASKS)), dtype=np.int32)

    print('done')

    

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base=ImpalaModel,
        base_kwargs={'recurrent': True,
                     'hidden_size': args.recurrent_hidden_size, 'gray_scale': args.gray_scale},
        epsilon_RPO=args.epsilon_RPO)
    actor_critic.to(device)

    # Training agent
    agent = algo.PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        num_tasks=args.num_processes,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay)

    # Rollout storage for agent
    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size, device=device)

    # Load previous model
    if (args.continue_from_epoch > 0) and args.save_dir != "":
        save_path = pathlib.Path(args.save_dir, args.env + '_ppo_seed_' + args.seed + '_maxEnt')
        actor_critic_weighs = torch.load(
            os.path.join(save_path, args.env_name + "-epoch-{}.pt".format(args.continue_from_epoch)),
            map_location=device)
        actor_critic.load_state_dict(actor_critic_weighs['state_dict'])
        agent.optimizer.load_state_dict(actor_critic_weighs['optimizer_state_dict'])

    #  logger = maxEnt_Logger(args.num_processes, max_reward_seeds, start_train_test, envs.observation_space.shape,
    #                        envs.observation_space.shape, actor_critic.recurrent_hidden_state_size, device=device)
    
    logger = Logger(args.num_processes, envs.observation_space.shape,
                            envs.observation_space.shape, actor_critic.recurrent_hidden_state_size, device=device)
    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.obs_full.copy_(obs)
    rollouts.obs_sum.copy_(torch.zeros_like(obs))

    obs_train = eval_envs_dic['train_eval'].reset()
    logger.obs['train_eval'].copy_(obs_train)
    logger.obs_sum['train_eval'].copy_(torch.zeros_like(obs_train))
    for i in range(args.num_processes):
        logger.obs_vec['train_eval'][i].append(obs_train[i])

    obs_test = eval_envs_dic['test_eval'].reset()
    logger.obs['test_eval'].copy_(obs_test)
    logger.obs_sum['train_eval'].copy_(torch.zeros_like(obs_test))
    for i in range(args.num_processes):
         logger.obs_vec['test_eval'][i].append(obs_test[i])

    # # TODO : Plot Crafter
    # fig = plt.figure(figsize=(20, 20))
    # columns = 5
    # rows = 5
    # for i in range(1, columns * rows + 1):
    #     fig.add_subplot(rows, columns, i)
    #     plt.imshow(rollouts.obs[0][i].transpose(0, 2))
    #     plt.savefig(logdir + '/fig.png')

    seeds = torch.zeros(args.num_processes, 1)
    beta = 0 # beta for rew = ext_rew + beta * int_rew
    num_env_steps = hps.num_env_steps['maxEnt']
    num_updates = int(
        num_env_steps) // args.num_steps // args.num_processes

    for j in range(args.continue_from_epoch, args.continue_from_epoch + num_updates):

        # Policy rollouts
        actor_critic.eval()
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, _, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step].to(device), rollouts.recurrent_hidden_states[step].to(device),
                    rollouts.masks[step].to(device))

            # Observe reward and next obs
            obs, ext_reward, done, infos = envs.step(action.squeeze().cpu().numpy())
            int_reward = np.zeros_like(ext_reward)

            diff_all = obs.unsqueeze(0) - rollouts.obs.to(device)

            for i in range(len(done)):
                if done[i] == 1:
                    rollouts.obs_sum[i] = torch.zeros_like(rollouts.obs_full[i])
                    rollouts.obs_full[i].copy_(obs[i])
                    rollouts.step_env[i] = 0
                else:
                    actual_step_env = int(max(0, rollouts.step_env[i] - args.num_buffer))

                    episode_start = int(step + 1 - rollouts.step_env[i] + actual_step_env)
                    diff = diff_all[max(0, episode_start):step+1][:, i, :, :]
                    if episode_start < 0:
                        if not len(diff):
                            diff = diff_all[args.num_steps + episode_start:args.num_steps][:, i, :, :]
                        else:
                            diff = torch.cat((diff, diff_all[args.num_steps + episode_start:args.num_steps][:, i, :, :].to(device)), dim=0)
                    if args.p_norm == 0:
                        diff = (1.0 * (diff.abs() > 1e-5)).sum(1)
                    neighbor_size = args.neighbor_size
                    if len(diff) < args.neighbor_size:
                        neighbor_size = len(diff)
                    int_reward[i] = diff.flatten(start_dim=1).norm(p=args.p_norm, dim=1).sort().values[int(neighbor_size-1)]


            # for i, info in enumerate(infos):
            #     seeds[i] = info["level_seed"]
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, torch.from_numpy(ext_reward + beta * int_reward).unsqueeze(1), masks, bad_masks, seeds, infos, obs) # need to change to - reward + beta * int_reward
            
        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1].to(device), rollouts.recurrent_hidden_states[-1].to(device),
                rollouts.masks[-1].to(device)).detach()

        actor_critic.train()
        gamma = hps.gamma[args.env_name]
        rollouts.compute_returns(next_value, use_gae=True, gamma=gamma, gae_lambda=args.gae_lambda)

        value_loss, action_loss, dist_entropy, _ = agent.update(rollouts)

        rollouts.after_update()

        rew_batch, done_batch = rollouts.fetch_log_data()
        logger.feed_train(rew_batch, done_batch[1:])

        # Save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0 or j == args.continue_from_epoch + num_updates - 1) and j > args.continue_from_epoch:
            torch.save({'state_dict': actor_critic.state_dict(), 'optimizer_state_dict': agent.optimizer.state_dict(),
                        'step': j}, os.path.join(logdir, args.env_name + "-epoch-{}.pt".format(j)))

        # # Evaluate agent on evaluation tasks
        # if ((args.eval_interval is not None and j % args.eval_interval == 0) or j == args.continue_from_epoch):
        #     actor_critic.eval()
        #     eval_dic_rew = {}
        #     eval_dic_int_rew = {}
        #     eval_dic_done = {}
        #     eval_dic_seeds = {}
        
        #     for eval_disp_name in EVAL_ENVS:
        #         eval_dic_rew[eval_disp_name], eval_dic_int_rew[eval_disp_name], eval_dic_done[eval_disp_name], \
        #         eval_dic_seeds[eval_disp_name] = evaluate_procgen_maxEnt_avepool_original_L2(actor_critic=actor_critic, eval_envs_dic=eval_envs_dic,
        #                                                                                   env_name=eval_disp_name, device=device,
        #                                                                                   steps=args.num_steps, logger=logger, num_buffer=args.num_buffer,
        #                                                                                   kernel_size=args.kernel_size,
        #                                                                                   stride=args.stride, deterministic=False, p_norm=args.p_norm, neighbor_size=args.neighbor_size)
        
        #     logger.feed_eval_test(eval_dic_int_rew['train_eval'], eval_dic_done['train_eval'], eval_dic_rew['train_eval'],
        #                                  eval_dic_int_rew['test_eval'], eval_dic_done['test_eval'], eval_dic_rew['test_eval'],
        #                                  eval_dic_seeds['train_eval'], eval_dic_seeds['test_eval'])

        ################################### Code from train PPO crafter ###########################################

        # Evaluate agent on evaluation tasks
        if ((args.eval_interval is not None and j % args.eval_interval == 0) or j == args.continue_from_epoch):
            actor_critic.eval()

            eval_test_ext_rew, eval_test_int_rew, eval_test_done, successes = evaluate_procgen(actor_critic, eval_envs_dic['test_eval'],
                'test_eval', device, args.num_steps, logger, args.num_buffer, deterministic=False, p_norm=args.p_norm, neighbor_size=args.neighbor_size)
            
            eval_test_rew = eval_test_ext_rew + beta * eval_test_int_rew

            logger.feed_eval(eval_test_rew, eval_test_done)
            total_successes = np.concatenate([total_successes, successes], axis=0) # total_successes = [total_successes ; successes]
            latest_epoch = j

        # Extract episode_statistics from logger
        if j % args.log_interval == 0:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            latest_eval_num_steps = (latest_epoch + 1) * args.num_processes * args.num_steps

            # Success_rate & score calc
            success_rate = 100 * np.mean(total_successes, axis=0)
            score = np.exp(np.mean(np.log(1 + success_rate))) - 1

            # Get eval stats
            eval_stats = {
                "success_rate": {k: v for k, v in zip(TASKS, success_rate)},
                "score": score,
            }

            # Print stats
            print(f"Epoch number {latest_epoch} / total_num_steps {latest_eval_num_steps}  Score stats")
            print(json.dumps(eval_stats, indent=2))
            # Uploading score and success_rates to wandb
            [wandb.log({f"success_rate/{task}": success_rate}, step=latest_eval_num_steps) for task, success_rate in
             eval_stats["success_rate"].items()]
            wandb.log({"score": score}, step=latest_eval_num_steps)

            ############################

            print('Iter {}, num timesteps {}, num training episodes {}, '
                  'dist_entropy {:.3f}, value_loss {:.3f}, action_loss {:.3f}\n'
                  .format(j, total_num_steps, logger.num_episodes, dist_entropy, value_loss, action_loss))
            episode_statistics = logger.get_episode_statistics()

            print(
                'Last {} training episodes: \n'
                'train mean/median reward {:.1f}/{:.1f},\n'
                'train min/max reward {:.1f}/{:.1f}\n'
                .format(args.num_processes,
                        episode_statistics['Rewards/mean_episodes']['train'],
                        episode_statistics['Rewards/median_episodes']['train'],
                        episode_statistics['Rewards/min_episodes']['train'],
                        episode_statistics['Rewards/max_episodes']['train']))

            print(
                'test mean/median reward {:.1f}/{:.1f},\n'
                'test min/max reward {:.1f}/{:.1f}\n'
                .format(episode_statistics['Rewards/mean_episodes']['test'],
                        episode_statistics['Rewards/median_episodes']['test'],
                        episode_statistics['Rewards/min_episodes']['test'],
                        episode_statistics['Rewards/max_episodes']['test']))

            log = [total_num_steps] + [episode_statistics['Rewards/mean_episodes']['train']] + [
                episode_statistics['Rewards/min_episodes']['train']] + [
                      episode_statistics['Rewards/max_episodes']['train']]
            log += [episode_statistics['Rewards/mean_episodes']['test']] + [
                episode_statistics['Rewards/min_episodes']['test']] + [
                       episode_statistics['Rewards/max_episodes']['test']]
            progresslog.loc[len(progresslog)] = log

            if (len(episode_statistics) > 0):
                for key, value in episode_statistics.items():
                    if isinstance(value, dict):
                        for key_v, value_v in value.items():
                            wandb.log({key + "/" + key_v: value_v},
                                      step=(j + 1) * args.num_processes * args.num_steps)

            with open(logdir + '/progress_{}_seed_{}.csv'.format(args.env_name, args.seed), 'w') as f:
                progresslog.to_csv(f, index=False)

    ################################### Code from train PPO crafter ###########################################

        # # Print some stats
        # if j % args.log_interval == 0:
        #     total_num_steps = (j + 1) * args.num_processes * args.num_steps
        #     print('Iter {}, num timesteps {}, num training episodes {}, '
        #           'dist_entropy {:.3f}, value_loss {:.3f}, action_loss {:.3f}\n'
        #           .format(j, total_num_steps, logger.num_episodes, dist_entropy, value_loss, action_loss))
        #     episode_statistics = logger.get_episode_statistics()
    
        #     print(
        #         'Last {} training episodes: \n'
        #         'train mean/median intrinsic reward {:.1f}/{:.1f},\n'
        #         'train min/max intrinsic reward {:.1f}/{:.1f}\n'
        #         .format(args.num_processes,
        #                 episode_statistics['Rewards/mean_episodes']['train_eval'], episode_statistics['Rewards/median_episodes']['train_eval'],
        #                 episode_statistics['Rewards/min_episodes']['train_eval'], episode_statistics['Rewards/max_episodes']['train_eval']))
    
        #     print(
        #         'train mean/median extrinsic reward {:.1f}/{:.1f},\n'
        #         'train min/max extrinsic reward {:.1f}/{:.1f}\n'
        #         .format(episode_statistics['Rewards/mean_episodes']['train_eval_ext'], episode_statistics['Rewards/median_episodes']['train_eval_ext'],
        #                 episode_statistics['Rewards/min_episodes']['train_eval_ext'], episode_statistics['Rewards/max_episodes']['train_eval_ext']))
        #     print(
        #         'test mean/median intrinsic reward {:.1f}/{:.1f},\n'
        #         'test min/max intrinsic reward {:.1f}/{:.1f}\n'
        #         .format(episode_statistics['Rewards/mean_episodes']['test'], episode_statistics['Rewards/median_episodes']['test'],
        #                 episode_statistics['Rewards/min_episodes']['test'], episode_statistics['Rewards/max_episodes']['test']))
    
        #     print(
        #         'test mean/median extrinsic reward {:.1f}/{:.1f},\n'
        #         'test min/max extrinsic reward {:.1f}/{:.1f}\n'
        #         .format(episode_statistics['Rewards/mean_episodes']['test_ext'], episode_statistics['Rewards/median_episodes']['test_ext'],
        #                 episode_statistics['Rewards/min_episodes']['test_ext'], episode_statistics['Rewards/max_episodes']['test_ext']))
    
        #     log = [total_num_steps] + [episode_statistics['Rewards/mean_episodes']['train_eval']] + [episode_statistics['Rewards/min_episodes']['train_eval']] + [episode_statistics['Rewards/max_episodes']['train_eval']]
        #     log += [episode_statistics['Rewards/mean_episodes']['train_eval_ext']] + [episode_statistics['Rewards/min_episodes']['train_eval_ext']] + [episode_statistics['Rewards/max_episodes']['train_eval_ext']]
        #     log += [episode_statistics['Rewards/mean_episodes']['test']] + [episode_statistics['Rewards/min_episodes']['test']] + [episode_statistics['Rewards/max_episodes']['test']]
        #     log += [episode_statistics['Rewards/mean_episodes']['test_ext']] + [episode_statistics['Rewards/min_episodes']['test_ext']] + [episode_statistics['Rewards/max_episodes']['test_ext']]
        #     progresslog.loc[len(progresslog)] = log
    
        #     with open(logdir + '/progress_{}_seed_{}.csv'.format(args.env_name, args.seed), 'w') as f:
        #         progresslog.to_csv(f, index=False)
    
    # Training done. Close and clean up
    for eval_disp_name in EVAL_ENVS:
        eval_envs_dic[eval_disp_name].close()


if __name__ == "__main__":
    main()
