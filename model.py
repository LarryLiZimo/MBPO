import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam

class ResidualMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, num_hidden_layer: int):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layer-1)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_hidden_layer-1)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.input_proj(x))
        for layer, norm in zip(self.layers, self.norms):
            x = x + F.silu(layer(norm(x)))
        return self.output_proj(self.final_norm(x))

class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, num_hidden_layer: int):
        # the action bound is [-1,+1]
        super().__init__()
        self.fc = ResidualMLP(obs_dim, action_dim * 2, hidden_dim, num_hidden_layer)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # obs: (batch, obs_dim)
        mean, logstd = self.fc(obs).chunk(2, dim=-1)
        logstd = logstd.clamp(-20, 2)

        normal = Normal(mean, logstd.exp())
        x = normal.rsample()
        action = torch.tanh(x)

        log_prob = normal.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    @torch.no_grad()
    def get_action(self, obs: torch.Tensor, sampling: bool):
        mean, logstd = self.fc(obs).chunk(2, dim=-1)
        logstd = logstd.clamp(-20, 2)
        if sampling:
            normal = Normal(mean, logstd.exp())
            return torch.tanh(normal.sample())
        else:
            return torch.tanh(mean)

class Critic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, num_hidden_layer: int):
        super().__init__()
        self.q1 = ResidualMLP(obs_dim + action_dim, 1, hidden_dim, num_hidden_layer)
        self.q2 = ResidualMLP(obs_dim + action_dim, 1, hidden_dim, num_hidden_layer)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # obs: (batch, obs_dim), action: (batch, action_dim)
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)


class SAC(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        actor_hidden_dim: int,
        actor_num_hidden_layer: int,
        critic_hidden_dim: int,
        critic_num_hidden_layer: int,
        actor_lr: float,
        critic_lr: float,
        alpha_lr: float,
        target_H: float,
        gamma: float,
        tau: float
    ):

        super().__init__()
        self.gamma = gamma
        self.tau = tau
        self.target_H = target_H

        self.actor = Actor(obs_dim, action_dim, actor_hidden_dim, actor_num_hidden_layer)
        self.critic = Critic(obs_dim, action_dim, critic_hidden_dim, critic_num_hidden_layer)
        self.critic_target = Critic(obs_dim, action_dim, critic_hidden_dim, critic_num_hidden_layer)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.log_alpha = nn.Parameter(torch.zeros(1))

        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
        self.alpha_optimizer = Adam([self.log_alpha], lr=alpha_lr)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor(obs)

    def update(self, obs: torch.Tensor, action: torch.Tensor, reward: torch.Tensor,
               next_obs: torch.Tensor, done: torch.Tensor) -> dict:
        assert obs.ndim == 2
        assert action.ndim ==2
        assert reward.ndim==1
        assert next_obs.ndim==2
        assert done.ndim==1
        alpha = self.log_alpha.exp().detach()

        with torch.no_grad():
            next_action, next_log_prob = self.actor(next_obs)
            q1_target, q2_target = self.critic_target(next_obs, next_action)
            q_target = torch.min(q1_target, q2_target) - alpha * next_log_prob
            backup = reward.unsqueeze(-1) + self.gamma * (1 - done.unsqueeze(-1)) * q_target

        q1, q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        new_action, log_prob = self.actor(obs)
        q1, q2 = self.critic(obs, new_action)
        actor_loss = (alpha * log_prob - torch.min(q1, q2)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_H).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": alpha.item(),
        }

class WorldModel(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, num_hidden_layer: int, lr: float):
        super().__init__()
        self.obs_dim = obs_dim
        self.fc = ResidualMLP(obs_dim + action_dim, obs_dim * 2 + 2, hidden_dim, num_hidden_layer)
        self.optimizer = Adam(self.parameters(), lr=lr)

    def update(self, *args, **kwargs) -> float:
        self.optimizer.zero_grad()
        out, loss = self.forward(*args, **kwargs)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def forward(self, obs, action, reward=None, next_obs=None,):
        B, _ = obs.shape
        assert (next_obs is None) == (reward is None)

        x = torch.cat([obs, action], dim=-1)
        mean, logstd = self.fc(x).chunk(2, dim=-1)
        logstd = logstd.clamp(-10, 0.5)

        dist = Normal(mean, logstd.exp())
        out = dist.rsample() + torch.cat([obs, torch.zeros(B, 1, device=obs.device)], dim=-1)

        loss = None
        if next_obs is not None:
            y = torch.cat([next_obs, reward.unsqueeze(-1)], dim=-1)
            pred_mean = mean + torch.cat([obs, torch.zeros(B, 1, device=obs.device, dtype=obs.dtype)], dim=-1)
            loss = torch.mean((pred_mean - y).pow(2) / (2 * logstd.exp().pow(2)) + logstd)

        return out, loss
