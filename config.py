from dataclasses import dataclass

@dataclass
class Config:
    env_name: str = "Pendulum-v1" # default para for Pendulum

    # SAC
    actor_hidden_dim: int = 64
    actor_num_hidden_layer: int = 1
    critic_hidden_dim: int = 64
    critic_num_hidden_layer: int = 1
    
    actor_lr: float = 3e-4
    critic_lr: float = 3e-3
    alpha_lr: float = 3e-3
    gamma: float = 0.99
    tau: float = 0.005
    target_entropy: float | None = None

    sac_batch_size: int = 64
    sac_gradient_step: int = 1


    # World Model
    wm_hidden_dim: int = 64
    wm_num_hidden_layer: int = 2

    wm_lr: float = 3e-3
    
    wm_rollout_interval: int = 1
    wm_rollout_batch_size: int = 64

    wm_train_interval: int = 1
    wm_train_batch_size: int = 64


    # Train
    max_step: int = 100_000
    D_env_capacity: int = 1000_000
    D_model_capacity: int = 1000_000 
    log_dir: str = "logs"
    save_dir: str = "checkpoints"
    checkpoint_interval: int = 10_000
    resume_id: str | None = None


# Requirements:
# 1. Each parameter must appear
# 2. The empty lines be the same as the Config class
CONFIG: list[Config] = [
    Config(
        env_name="HalfCheetah-v4",

        # SAC
        actor_hidden_dim=256,
        actor_num_hidden_layer=2,
        critic_hidden_dim=256,
        critic_num_hidden_layer=2,
        
        actor_lr=3e-4,
        critic_lr=3e-3,
        alpha_lr=3e-4,
        gamma=0.99,
        tau=0.005,
        target_entropy=None,

        sac_batch_size=256,
        sac_gradient_step=4,


        # World Model
        wm_hidden_dim=256,
        wm_num_hidden_layer=3,

        wm_lr=1e-3,
        
        wm_rollout_interval=5,
        wm_rollout_batch_size=256,

        wm_train_interval=1,
        wm_train_batch_size=256,


        # Train
        max_step=300_000,
        D_env_capacity=1_000_000,
        D_model_capacity=400_000,
        log_dir="logs",
        save_dir="checkpoints",
        checkpoint_interval=10_000,
        resume_id=None,
    )
]

