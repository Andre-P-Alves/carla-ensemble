import torch.optim as optim
import numpy as np
import tqdm
import torch as th
from pathlib import Path
import wandb
import gym
from rl_train import get_obs_configs, get_env_wrapper_configs, env_maker 
from stable_baselines3.common.vec_env import SubprocVecEnv
from rl_birdview.utils.wandb_callback import WandbCallback
from rl_birdview.models.discriminator import ExpertDataset
from rl_birdview.models.ppo_policy import PpoPolicy
from rl_birdview.models.discriminator_copy import ExpertDataset as RandomSampler
import matplotlib.pyplot as plt


def learn_bc(policy, device, expert_loader, eval_loader, n_ensemble):
    bc_losses = []
    bc_steps = []
    evaluation_losses = []
    evaluation_steps = []
    bc2_losses = []
    bc2_steps = []
    output_dir = Path('outputs')
    output_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint_path = output_dir / 'checkpoint.txt'

    ckpt_dir = Path('ckpt')
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    video_path = Path('video')
    video_path.mkdir(parents=True, exist_ok=True)

    optimizer = optim.Adam(policy.parameters(), lr=1e-5)
    max_grad_norm = 0.5
    episodes = 200
    ent_weight = 0.01
    min_eval_loss = np.inf
    eval_step = int(1e5)
    steps_last_eval = 0
    expert_fake_birdview_loader = policy.gan_fake_birdview.fill_expert_dataset(expert_loader)
    start_ep = 0
    i_steps = 0
    for i_episode in tqdm.tqdm(range(start_ep, episodes)):
        total_loss = 0
        i_batch = 0
        policy = policy.train()
        # Expert dataset
        for expert_batch in expert_loader:
            expert_obs_dict, expert_action = expert_batch
            obs_tensor_dict = {
                'state': expert_obs_dict['state'].float().to(device),
                'birdview': expert_obs_dict['birdview'].float().to(device)
            }
            fake_birdview = expert_fake_birdview_loader.index_select(dim=0, index=expert_obs_dict['item_idx'].int())
            fake_birdview = fake_birdview.to(policy.device)
            expert_action = expert_action.to(device)

            # Get BC loss
            alogprobs, entropy_loss = policy.evaluate_actions_bc(obs_tensor_dict, fake_birdview, expert_action)
            bcloss = -alogprobs.mean()

            loss = bcloss + ent_weight * entropy_loss
            bc_losses.append(loss)
            bc_steps.append(i_batch)
            total_loss += loss
            i_batch += 1
            i_steps += expert_obs_dict['state'].shape[0]
            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_eval_loss = 0
        i_eval_batch = 0
        eval_fake_birdview_loader = policy.gan_fake_birdview.fill_expert_dataset(expert_loader)
        for expert_batch in eval_loader:
            expert_obs_dict, expert_action = expert_batch
            obs_tensor_dict = {
                'state': expert_obs_dict['state'].float().to(device),
                'birdview': expert_obs_dict['birdview'].float().to(device)
            }
            fake_birdview = eval_fake_birdview_loader.index_select(dim=0, index=expert_obs_dict['item_idx'].int())
            fake_birdview = fake_birdview.to(policy.device)
            expert_action = expert_action.to(device)

            # Get BC loss
            with th.no_grad():
                alogprobs, entropy_loss = policy.evaluate_actions_bc(obs_tensor_dict, fake_birdview, expert_action)
            bcloss = -alogprobs.mean()
            
            evaluation_losses.append(eval_loss)
            evaluation_steps.append(i_eval_batch)

            bc2_losses.append(bcloss)
            bc2_steps.append(i_eval_batch)

            eval_loss = bcloss + ent_weight * entropy_loss
            total_eval_loss += eval_loss
            i_eval_batch += 1
            
        
        loss = total_loss / i_batch
        eval_loss = total_eval_loss / i_eval_batch
        '''wandb.log({
            'loss': loss,
            'eval_loss': eval_loss,
        }, step=i_steps)'''

        if min_eval_loss > eval_loss:
            ckpt_path = (ckpt_dir / f'bc_ckpt_{i_episode}_min_eval.pth').as_posix()
            th.save(
                {'policy_state_dict': policy.state_dict()},
               ckpt_path
            )
            min_eval_loss = eval_loss

        train_init_kwargs = {
            'start_ep': i_episode,
            'i_steps': i_steps
        } 
        ckpt_path = (ckpt_dir / 'ckpt_latest.pth').as_posix()
        th.save({'policy_state_dict': policy.state_dict(),
                 'train_init_kwargs': train_init_kwargs},
                ckpt_path)
        #wandb.save(f'./{ckpt_path}')
    print("Finished")
    run = run.finish()

    data_sets = [
        {'y': bc_losses, 'x': bc_steps, 'title': 'BC1'},
        {'y': bc2_losses, 'x': bc2_steps, 'title': 'BC2'},
        {'y': evaluation_losses, 'x': evaluation_steps, 'title': 'Evaluation'}
    ]

    #plots
    for i, data in enumerate(data_sets):
        plt.plot(data['x'], data['y'])
        plt.xlabel('Batches')
        plt.ylabel('Loss')
        plt.title(data['title'])
        plt.savefig(f'ensemble{n_ensemble}_plot_{i+1}.png')
        plt.savefig(f'ensemble{n_ensemble}_plot_{i+1}.eps', format='eps')
        plt.clf()  

    th.save({'policy_state_dict': policy.state_dict(),
                 'train_init_kwargs': train_init_kwargs},
                ckpt_path+f'{n_ensemble}')

def train_gan(gan_fake_birdview, train_dataloader):
    gan_disc_losses, gan_generator_losses, gan_pixel_losses, gan_losses = [], [], [], []
    for i, batch in enumerate(train_dataloader):
        obs_dict, _ = batch
        gan_disc_loss, gan_generator_loss, gan_pixel_loss, gan_loss = gan_fake_birdview.train_batch(obs_dict)
        gan_disc_losses.append(gan_disc_loss)
        gan_generator_losses.append(gan_generator_loss)
        gan_pixel_losses.append(gan_pixel_loss)
        gan_losses.append(gan_loss)

    train_debug = {
        "train_gan/gan_disc_loss": np.mean(gan_disc_losses),
        "train_gan/gan_generator_loss": np.mean(gan_generator_losses),
        "train_gan/gan_pixel_loss": np.mean(gan_pixel_losses),
        "train_gan/gan_loss": np.mean(gan_losses)
    }

    return train_debug


def val_gan(gan_fake_birdview, val_dataloader):
    gan_disc_losses, gan_generator_losses, gan_pixel_losses, gan_losses = [], [], [], []
    for i, batch in enumerate(val_dataloader):
        obs_dict, _ = batch
        gan_disc_loss, gan_generator_loss, gan_pixel_loss, gan_loss = gan_fake_birdview.val_batch(obs_dict)
        gan_disc_losses.append(gan_disc_loss)
        gan_generator_losses.append(gan_generator_loss)
        gan_pixel_losses.append(gan_pixel_loss)
        gan_losses.append(gan_loss)

    val_debug = {
        "val_gan/gan_disc_loss": np.mean(gan_disc_losses),
        "val_gan/gan_generator_loss": np.mean(gan_generator_losses),
        "val_gan/gan_pixel_loss": np.mean(gan_pixel_losses),
        "val_gan/gan_loss": np.mean(gan_losses)
    }

    return val_debug


if __name__ == '__main__':
    # network

    resume_last_train = False

    observation_space = {}
    observation_space['birdview'] = gym.spaces.Box(low=0, high=255, shape=(3, 192, 192), dtype=np.uint8)
    observation_space['state'] = gym.spaces.Box(low=-10.0, high=30.0, shape=(6,), dtype=np.float32)
    observation_space = gym.spaces.Dict(**observation_space)

    action_space = gym.spaces.Box(low=np.array([0, -1]), high=np.array([1, 1]), dtype=np.float32)


    policy_kwargs = {
        #'observation_space': env.observation_space,
        #'action_space': env.action_space,
        'observation_space': observation_space,
        'action_space': action_space,
        'policy_head_arch': [256, 256],
        'value_head_arch': [256, 256],
        'features_extractor_entry_point': 'rl_birdview.models.torch_layers:XtMaCNN',
        'features_extractor_kwargs': {'states_neurons': [256,256]},
        'distribution_entry_point': 'rl_birdview.models.distributions:BetaDistribution',
        'fake_birdview': False
    }

    device = 'cuda'

    policy = PpoPolicy(**policy_kwargs)
    policy.to(device)

    batch_size = 32

    gail_train_loader = th.utils.data.DataLoader(
        RandomSampler(
            'gail_experts',
            n_routes=1,
            n_eps=1,
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    gail_val_loader = th.utils.data.DataLoader(
        ExpertDataset(
            'gail_experts',
            n_routes=1,
            n_eps=1,
            route_start=1
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    for i in range(0,4):
        learn_bc(policy, device, gail_train_loader, gail_val_loader, i)
