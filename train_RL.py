# actor-critic network for Reinforcement Learning
import torch
from torch import nn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
from torch.autograd import Variable

import argparse
import math
import os
import random
from torch.distributions import Categorical
from varname import nameof

from models.models import *
from utils.LR_utils import normalize, denormalize, get_state, get_state2, get_prunable_layers_yolov4, get_layers_forpruning, list2FloatTensor, \
    test_alpha_seq
from models.LR_models import actorNet, criticNet, actorNet2, init_weights
from utils.RL_rewards import reward_function
from models.error_pred_network import errorNet
from utils.LR_losses import CriticLoss, ActorLoss, ActorPPOLoss, get_discounted_reward, get_advantage, \
    get_discounted_reward
# from utils.state_tester import get_fix_state
from utils.RL_logger import RLLogger, TensorboardLogger
from utils.torch_utils import init_seeds
from utils.optimizers import RAdam, Lamb
from utils.config_parser import ConfigParser



timefile = "/home/blanka/YOLOv4_Pruning/sandbox/time_measure_pruning3.txt"

if __name__ == '__main__':

    
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default="rl_agent")
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--test-case', type=str, default='trans_spn6_04a')

    # Flags
    parser.add_argument('--variable_logflag', type=bool, default=True)
    parser.add_argument('--lr_sched_step_flag', type=bool, default=False)
    parser.add_argument('--set_new_lr', type=bool, default=True)
    parser.add_argument('--set-new-lossfunc', type=bool, default=False)
    parser.add_argument('--test', type=bool, default=False)

    opt = parser.parse_args()

    conf = ConfigParser.prepare_conf(opt)
    if len(opt.device):
        device = opt.device
    else:
        device = conf.train.device

    torch.autograd.set_detect_anomaly(True)
    init_seeds(42)

    # Create logger
    rl_logger = RLLogger(log_dir=os.path.join(conf.paths.log_dir, conf.logging.folder), test_case=opt.test_case)
    tb_logger = TensorboardLogger(log_dir=os.path.join(conf.paths.log_dir, conf.logging.folder), test_case=opt.test_case)

    # Load pretrained nets
    net_for_pruning = Darknet(conf.models.cfg_to_prune).to(device)
    ckpt_nfp = torch.load(os.path.join(conf.paths.model_dir, conf.models.to_prune))
    state_dict = {k: v for k, v in ckpt_nfp['model'].items() if net_for_pruning.state_dict()[k].numel() == v.numel()}
    net_for_pruning.load_state_dict(state_dict, strict=False)
    network_size = len(net_for_pruning.module_list)

    # Load pretrained SPN
    ckpt_spn = torch.load(os.path.join(conf.paths.model_dir, conf.models.spn), map_location=device)
    spn = ckpt_spn['model']
    spn.eval()

    # Define alpha values
    alphas = np.arange(0.0, 2.3, 0.1).tolist()
    alphas = [float("{:.2f}".format(x)) for x in alphas]

    # Initialize actor and critic networks

    if len(conf.models.rl_pretrained):
        ckpt = torch.load(opt.pretrained)
        actorNet = ckpt['actor_model']
        print(actorNet)
        criticNet = ckpt['critic_model']
        actor_optimizer = ckpt['actor_optimizer']
        critic_optimizer = ckpt['critic_optimizer']
        episode = ckpt['episode']
        lr_sched = ckpt['lr_sched']
        log_probs_prev = ckpt['log_probs']

        # Change learning rate
        if opt.set_new_lr:
            for g in actor_optimizer.param_groups:
                g['lr'] = conf.a2c.actor_base_lr
            lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(actor_optimizer, T_max=conf.train.episodes,
                                                                  eta_min=conf.a2c.actor_last_lr,
                                                                  last_epoch=episode)
            lr_sched.step()

        critic_criterion = ckpt['critic_criterion']
        actor_criterion =  ckpt['actor_criterion']
        print("pretrained", episode)
        eps = 3.964
    else:
        print("new model")

        actorNet = actorNet2(conf.prune.n_prunable_layers * 6, len(alphas)).to(device)
        #actorNet.apply(init_weights)
        criticNet = criticNet(conf.prune.n_prunable_layers * 6, 1).to(device)
        if conf.a2c.optim == 'adam':
            actor_optimizer = torch.optim.Adam(actorNet.parameters(), lr=conf.a2c.actor_base_lr)
            critic_optimizer = torch.optim.Adam(criticNet.parameters(), lr=conf.a2c.critic_base_lr)
        elif conf.a2c.optim == 'lamb':
            actor_optimizer = Lamb(actorNet.parameters(), lr=conf.a2c.actor_base_lr, weight_decay=1e-5)
            critic_optimizer = Lamb(criticNet.parameters(), lr=conf.a2c.critic_base_lr, weight_decay=1e-5)  

        lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(actor_optimizer, T_max=conf.train.episodes,
                                                            eta_min=conf.a2c.actor_last_lr, last_epoch=-1)

        # Define loss functions
        critic_criterion = CriticLoss().to(device)
        actor_criterion = ActorLoss().to(device)
        episode = 0

    # Print the number of params od actor and critic nets
    actor_total_params = sum(param.numel() for param in actorNet.parameters())
    critic_total_params = sum(param.numel() for param in criticNet.parameters())
    print(f"actor, critic params in main: {actor_total_params}, {critic_total_params}")

    # Get layer indicies that can be pruned
    layers_for_pruning = get_layers_forpruning(net_for_pruning, conf.prune.layers_to_skip)
    print(f"{len(layers_for_pruning) = }")


    # Log settings
    settings_dict = {"episode": episode,
                     "actor_lr": actor_optimizer.param_groups[0]['lr'],
                     "critic_lr": critic_optimizer.param_groups[0]['lr'],
                     "actor_lr_sched": lr_sched.get_lr()[0]
                     }
    rl_logger.log_settings(opt, settings_dict)

    while episode < conf.train.episodes:
        start_time_episode = time.time()

        network_seq = []
        # Deepcopy of the initialized network
        # for ni in range(opt.batch_size):
        #    network_seq.append(pickle.loads(pickle.dumps(net_for_pruning)))
        init_param_nmb = sum([param.nelement() for param in net_for_pruning.parameters()])

        action_seq = torch.full([conf.train.batch_size, 1, conf.prune.n_prunable_layers], -1.0)
        state_seq = torch.full([conf.train.batch_size, 6, conf.prune.n_prunable_layers], -1.0)

        actions = []
        states = []
        rewards = []
        rewards_list = []
        values = []
        policies = []
        log_probs = []
        entropies = []
        errors = []
        dEs, dSs = [], []

        layer_cnt = 0
        sparsity_prev = torch.full([conf.train.batch_size], -1.0)
        error_prev = torch.full([conf.train.batch_size], -1.0)

        for layer_i in range(network_size):

            if layer_i in layers_for_pruning:  
                # print("Pruning layer ", layer_cnt, layer_i)
                sequential_size = len(net_for_pruning.module_list[layer_i])
                layer = [net_for_pruning.module_list[layer_i][j] for j in range(sequential_size) if
                         isinstance(net_for_pruning.module_list[layer_i][j], nn.Conv2d)]
                layer = layer[0]

                # Get state
                # state_seq = Variable(get_state2(state_seq, sparsity_prev, layer, layer_cnt), requires_grad=True)
                state_seq = get_state2(state_seq, sparsity_prev, layer, layer_cnt)

                data = state_seq.view([conf.train.batch_size, -1]).type(torch.float32).to(device)

                print(f"data in main {data.shape}")

                probs, action_dist, log_softmax = actorNet(data)
                q_value = criticNet(data)
                # print("dist", action_dist)
                # print("value", q_value)

                # Choose alpha values based on probabilities
                action = action_dist.sample()  # alpha index
                # entropy = action_dist.entropy()
                log_prob = action_dist.log_prob(action).unsqueeze(1)
                policy = probs.gather(-1, action.unsqueeze(0))
                entropy = - (probs * log_softmax).sum(1, keepdim=True)

                # Log probs for the first and last layer
                if layer_cnt == 0 or layer_cnt == conf.prune.n_prunable_layers - 1:
                    rl_logger.log_probs(probs, episode, layer_cnt)
                    tb_logger.log_probs_merged(probs, episode, layer_cnt)

                tb_logger.log_probs(probs, episode, layer_cnt)

                for i in range(conf.train.batch_size):
                    action_seq[i, :, layer_cnt] = normalize(alphas[action[i]], 0.0, 2.2)

                print(denormalize(action_seq[0, :, :], 0.0, 2.2))

                # Get the error for every sample in the batch
                spn_input_data = torch.cat((action_seq, state_seq[:, -1, :].unsqueeze(1)), dim=1).view(
                    [conf.train.batch_size, -1]).type(torch.float32).to(device)
                prediction = spn(spn_input_data)
                error, sparsity = prediction[:,0], prediction[:,1]
                errors.append(error.detach().unsqueeze(1))
                
                reward = reward_function(denormalize(error, 0, 1), conf.reward.target_error, denormalize(sparsity, 0, 1),
                                               conf.reward.target_spars, conf.reward.err_coef, conf.reward.spars_coef, device,
                                               conf.reward.beta)  

                reward = reward.unsqueeze(1)
                ## rewards_list.append(reward)
                # print("rewarrrd ", reward)

                # if prediction[0,0]<0:7
                if False:
                    print(errorNet_input_data[0])
                    print("prediction", prediction[0])
                    print("reward ", reward[0])
                    print(log_prob[0])

                # if layer_cnt == 43:
                #    print("inal_spars", sparsity)
                #    print("final reward ", reward)

                layer_cnt += 1
                sparsity_prev = sparsity.clone()
                error_prev = error.clone()

                # Save the trajectory
                # log_probs.append(log_prob.clone().detach())
                log_probs.append(log_prob)  # .detach())

                entropies.append(entropy)  # .detach())
                ## list2FloatTensor(entropies)
                actions.append(action_seq.clone().detach())
                states.append(state_seq.clone().detach())
                # rewards.append(denormalize(reward, 0, 1))
                # values.append(denormalize(q_value, 0, 1))
                rewards.append(reward)  # .detach())
                values.append(q_value)  # .detach())
                policies.append(policy)  # .detach())
                # dEs.append(dE.clone().detach().unsqueeze(1))
                # dSs.append(dS.clone().detach().unsqueeze(1))

                # Diagnostics
                print(
                    f"reward {reward.shape} {reward.device}, log_prob: {log_prob.shape} {log_prob.device}, values: {q_value.shape} {q_value.device}, policy: {policy.shape} {policy.device}, entropies {entropy.shape}")

        # Test the results
        if opt.test:
            test_alpha_seq(denormalize(list2FloatTensor(errors)[-1, :, 0], 0, -1),
                           denormalize(states[-1][:, -1, -1], 0, 1),
                           list2FloatTensor(rewards_list)[-1, :, 0],
                           denormalize(actions[-1][:, 0, :], 0, 2.2),
                           conf.prune.layers_to_skip,
                           test_case="test_58_d_2700",
                           error_thresh=None, spars_thresh=None, reward_thresh=-5)
            
        
        	
        # Get the best result from the batch
        print(len(states))
        print(denormalize(states[-1][:, -1, -1], 0, 1).shape)
        print(denormalize(actions[-1][:, :, :], 0, 2.2).shape)

        #print(denormalize(list2FloatTensor(errors)[-1, :, :], 0, -1).shape)
        #print(denormalize(states[-1][:, -1, -1], 0, 1).shape)
        #print(denormalize(actions[-1][:, :, :], 0, 2.2).shape)

        bidx = denormalize(list2FloatTensor(errors)[-1, :, 0], 0, 1).argmin() # [n_prunableLayers, batch_size, 1]
        best_error = denormalize(list2FloatTensor(errors)[-1, bidx, 0], 0, 1).item()
        best_spars = denormalize(states[-1][bidx, -1, -1], 0, 1).item()
        best_alpha_seq = denormalize(actions[-1][bidx, 0, :], 0, 2.2)


        returns = get_discounted_reward(list2FloatTensor(rewards), list2FloatTensor(values), gamma=0.99)

        if episode == 0:
            print(episode, "notlenlogprob")
            log_probs_prev = torch.zeros(list2FloatTensor(log_probs).shape)
        else:
            # print(len(log_probs_prev), log_probs_prev[0].shape)
            # print(len(log_probs), log_probs[0].shape)
            log_probs_prev = list2FloatTensor(log_probs_prev).detach()

        # print("prev", log_probs_prev[0])
        # print("cur", log_probs[0])

        # Loss backwards here
        critic_loss = critic_criterion(list2FloatTensor(rewards), list2FloatTensor(values), 0.99)
        # critic_loss.backward(retain_graph=True)      
        actor_loss = actor_criterion(list2FloatTensor(rewards), list2FloatTensor(values),
                                        list2FloatTensor(policies), list2FloatTensor(log_probs),
                                        list2FloatTensor(entropies), ent_coef=conf.a2c.ent_coef, gamma=0.99)


        ## log_probs_prev = [log_prob.detach() for log_prob in log_probs]

        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()

        final_loss = actor_loss + critic_loss
        final_loss.backward(retain_graph=True)

        # print(f"Reward device: {rewards[0].device}")
        # break
        reward_backprop = list2FloatTensor(rewards).mean()
        (-reward_backprop).backward()

        actor_optimizer.step()
        critic_optimizer.step()
        if opt.lr_sched_step_flag: lr_sched.step()

        print(critic_loss)
        print(actor_loss)

        ## critic_loss.detach_()
        ## actor_loss.detach_()
        ## final_loss.detach_()

        # LOGGING

        if opt.variable_logflag:

            # print(actions[0])
            rl_logger.log_variables(denormalize(actions[-1][:, 0, :], 0, 2.2), nameof(actions), episode=None)
            rl_logger.log_variables(list2FloatTensor(values), nameof(values), episode=None)
            ## rl_logger.log_variables(denormalize(list2FloatTensor(errors)[:, :, 0], 0, 1), nameof(error), episode=None)
            ## rl_logger.log_variables(denormalize(states[-1][:, -1, :], 0, 1), nameof(sparsity), episode=None)
            ## rl_logger.log_variables(list2FloatTensor(rewards_list)[:, :, 0], nameof(rewards_list), episode=None)
            rl_logger.log_variables(returns, nameof(returns), episode=None)

            ## tb_logger.log_variables(episode, denormalize(states[-1][:, -1, :], 0, 1), nameof(sparsity), dim=0)
            tb_logger.log_variables(episode, denormalize(actions[-1][:, 0, :], 0, 2.2), nameof(actions), dim=0)
            ## tb_logger.log_variables(episode, denormalize(list2FloatTensor(errors)[:, :, 0], 0, 1), nameof(error), dim=1)
            ## tb_logger.log_variables(episode, list2FloatTensor(rewards_list)[:, :, 0], nameof(rewards_list), dim=1)

            # tb_logger.log_variables(episode, list2FloatTensor(dEs)[:,:,0], nameof(dE), dim=1)
            # tb_logger.log_variables(episode, list2FloatTensor(dSs)[:,:,0], nameof(dS), dim=1)

        # Save parameters
        rl_logger.log_results(episode, critic_loss.item(), actor_loss.item(),
                              list2FloatTensor(rewards).mean().item())
        rl_logger.log_learning_rate(episode, lr_sched)
        rl_logger.log_bests(episode, best_error, best_spars, best_alpha_seq)
        tb_logger.log_results(episode, critic_loss.item(), actor_loss.item(), list2FloatTensor(rewards).mean().item())
        tb_logger.log_learning_rate(episode, lr_sched)
        tb_logger.log_bests(episode, best_error, best_spars)

        # Save checkpoint
        checkpoint = {'episode': episode,
                      'actor_model': actorNet,
                      'critic_model': criticNet,
                      'actor_optimizer': actor_optimizer,
                      'critic_optimizer': critic_optimizer,
                      'actor_criterion': actor_criterion,
                      'critic_criterion': critic_criterion,
                      'lr_sched': lr_sched,
                      'log_probs': log_probs_prev
                      }

        # Save the checkpoint with episode
        if episode % conf.train.save_interval == 0:
            ckp_save_path = os.path.join(conf.paths.log_dir, conf.logging.folder, opt.test_case)
            torch.save(checkpoint, os.path.join(ckp_save_path, f"{episode}.pth)"))
            rl_logger.save_action_csv(episode, actions, errors, states)


        episode += 1
        print("Episode time ", time.time() - start_time_episode)