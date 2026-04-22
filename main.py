import os
import numpy as np
import torch
import gymnasium as gym
import tyro
import datetime
from torch import from_numpy
from torch.utils.tensorboard import SummaryWriter

from config import Config, CONFIG
from model import Actor, SAC, WorldModel
from replay_buffer import ReplayBuffer


def eval_policy(env: gym.Env, actor: Actor, num_episode: int, device: torch.device) -> float:
    actor.eval()
    total_reward = 0.0

    for _ in range(num_episode):
        obs, info = env.reset()
        done = False

        while not done:
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).float().to(device)
                action = actor.get_action(obs_tensor, sampling=False).cpu().numpy().squeeze(0)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

    actor.train()
    return total_reward / num_episode

def fuck(*args, device: torch.device):
    return [from_numpy(arg).to(device) for arg in args]


def main(cfg:Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(cfg.log_dir, f'{cfg.env_name}-{run_id}')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(cfg.save_dir, exist_ok=True)

    env = gym.make(cfg.env_name)
    eval_env = gym.make(cfg.env_name)
    env = gym.wrappers.RescaleAction(env, -1.0, 1.0)
    eval_env = gym.wrappers.RescaleAction(eval_env, -1.0, 1.0)
    env = gym.wrappers.TransformObservation(env, lambda obs: obs.astype(np.float32), env.observation_space)
    eval_env = gym.wrappers.TransformObservation(eval_env, lambda obs: obs.astype(np.float32), eval_env.observation_space)

    assert isinstance(env.observation_space, gym.spaces.Box), "Only Box observation spaces are supported."
    assert isinstance(env.action_space, gym.spaces.Box), "Only Box action spaces are supported."
   

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    sac = SAC(
        obs_dim=obs_dim,
        action_dim=action_dim,
        actor_hidden_dim=cfg.actor_hidden_dim,
        actor_num_hidden_layer=cfg.actor_num_hidden_layer,
        critic_hidden_dim=cfg.critic_hidden_dim,
        critic_num_hidden_layer=cfg.critic_num_hidden_layer,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        alpha_lr=cfg.alpha_lr,
        target_H=cfg.target_entropy if cfg.target_entropy is not None else -float(action_dim),
        gamma=cfg.gamma,
        tau=cfg.tau,
    ).to(device)

    wm = WorldModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=cfg.wm_hidden_dim,
        num_hidden_layer=cfg.wm_num_hidden_layer,
        lr=cfg.wm_lr,
    ).to(device)

    D_env = ReplayBuffer(obs_dim, action_dim, cfg.D_env_capacity)
    D_model = ReplayBuffer(obs_dim, action_dim, cfg.D_model_capacity)

    step = 0
    cumulative_reward = 0.0
    if cfg.resume_id is not None:
        resume_ckpt_path = os.path.join(cfg.save_dir, f"{cfg.env_name}-{cfg.resume_id}.pt")
        checkpoint = torch.load(resume_ckpt_path, map_location=device, weights_only=False)
        sac.load_state_dict(checkpoint["sac"])
        wm.load_state_dict(checkpoint["wm"])

    writer = SummaryWriter(log_dir=log_dir)

    print(f'{cfg.env_name = }')
    print(f"Initialized MBPO for {cfg.env_name} on {device}.")
    print(f"obs_dim={obs_dim}, action_dim={action_dim}")
    print(f"Started new run {run_id}{f' from weights in {cfg.resume_id}' if cfg.resume_id is not None else ''}.")

    obs, info = env.reset()

    while step < cfg.max_step:

        # # Step 1: wm rollout
        if step % cfg.wm_rollout_interval == 0 and D_env.can_sample(cfg.wm_rollout_batch_size):
            sac.eval()
            wm.eval()
            fobs, action, reward, next_obs, done = D_env.sample(cfg.wm_rollout_batch_size)
            fobs = from_numpy(fobs).to(device)
            with torch.no_grad():
                action = sac.actor.get_action(fobs, sampling=True)
                out, loss = wm(fobs, action)
                next_obs, reward = out[:, :-1], out[:, -1]
                D_model.add_batch(fobs.cpu().numpy(), action.cpu().numpy(), reward.cpu().numpy(), next_obs.cpu().numpy(), np.zeros_like(reward.cpu().numpy()))
           
        
        # # Step 2: wm train
        if step % cfg.wm_train_interval == 0 and D_env.can_sample(cfg.wm_train_batch_size):
            wm.train()
            # slice out obs, action, next_obs, reward
            wm_loss = wm.update(*fuck(*(D_env.sample(cfg.wm_train_batch_size)[:4]), device=device))
            writer.add_scalar("train/wm_loss", wm_loss, step)

        # Step 3: env rollout
        sac.eval()
        action = sac.actor.get_action(from_numpy(obs).unsqueeze(0).to(device), sampling=True).cpu().squeeze(0).numpy()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        step+=1
        cumulative_reward+=reward
        D_env.add(obs, action, reward, next_obs, terminated or truncated)
        if terminated or truncated:
            obs, info = env.reset()
            r = eval_policy(eval_env, sac.actor, 10, device)
            print(f"Step {step}; Reward: {r:.2f}; TrainReward:{cumulative_reward:.2f}")
            writer.add_scalar("eval/return", r, step)
            writer.add_scalar("train/episode_return", cumulative_reward, step)
            cumulative_reward = 0.0
        else:
            obs = next_obs
        
        # Step 4: sac train
        sac.train()
        if D_model.can_sample(cfg.sac_batch_size):
            sac_loss_dict = {}
            for g in range(cfg.sac_gradient_step):
                loss_dict = sac.update(*fuck(*D_model.sample(cfg.sac_batch_size), device=device))
                for k, v in loss_dict.items():
                    sac_loss_dict[k] = sac_loss_dict.get(k, 0) + v
            for k, v in sac_loss_dict.items():
                writer.add_scalar(f"train/{k}", v, step)
        
        # Step 5: checkpoint save
        if step % cfg.checkpoint_interval == 0:
            torch.save(
                {"sac": sac.state_dict(), "wm": wm.state_dict()},
                os.path.join(cfg.save_dir, f"{cfg.env_name}-{run_id}.pt"),
            )

    writer.close()

if __name__ == "__main__":
    # main(tyro.cli(Config))
    main(CONFIG[0])
