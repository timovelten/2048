import torch as th
from torch import nn
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.torch_layers import FlattenExtractor

def build_fc_layers(input_size, layer_sizes, activation_fn=nn.ReLU, last_layer_has_activation_fn=True, device=None):
    output_size = input_size
    layers = []
    for i, layer_size in enumerate(layer_sizes):
        layers.append(nn.Linear(output_size, layer_size, device=device))

        if i < len(layer_sizes) - 1 or last_layer_has_activation_fn:
            layers.append(activation_fn())
            
        output_size = layer_size
    return layers, output_size

class BaseModel(nn.Module):
    def __init__(self, 
                 shared_net: th.nn.Module,
                 action_space: spaces.Discrete,  
                 vf_net: th.nn.Module,
                 vf_net_features: int,
                 pi_net: th.nn.Module,
                 pi_net_features: int,
                 device = None):
        super().__init__()
        self.device = device
        self.action_space = action_space
        self.shared_net = shared_net
        self.vf_net = vf_net
        self.vf_head = nn.Linear(vf_net_features, 1, device=device)

        self.pi_net = pi_net
        self.pi_head = nn.Linear(pi_net_features, int(action_space.n), device=device)

    def obs_as_tensor(self, obs):
        return th.as_tensor(obs, dtype=th.float, device=self.device)

    def forward_critic(self, obs: th.Tensor) -> th.Tensor:
        x = self.shared_net(obs)
        x = self.vf_net(x)
        return self.vf_head(x).squeeze()

    def forward(self, obs: th.Tensor, action_mask: np.ndarray | th.Tensor | None = None):
        x = self.shared_net(obs)
        pi = self.pi_head(self.pi_net(x))
        vf = self.vf_head(self.vf_net(x))

        if action_mask is not None:
            pi = th.where(th.as_tensor(action_mask, device=self.device), pi, -th.inf)
        
        return th.distributions.Categorical(logits=pi), vf.squeeze()
    
    def get_actions(self, obs, action_masks):
        with th.no_grad():
            pi, _ = self.forward(
                self.obs_as_tensor(obs), action_mask=action_masks
            )
            actions: np.ndarray = pi.mode.numpy(force=True)

        return actions

class DefaultModel(BaseModel):
    def __init__(self, 
                 observation_space: spaces.Box, 
                 action_space: spaces.Discrete, 
                 device = None,
                 pi_layers = [64, 64], 
                 vf_layers = [64, 64],
            ):
        shared_net = FlattenExtractor(observation_space)
        pi_layers, pi_net_features = build_fc_layers(shared_net.features_dim, pi_layers, device=device)
        vf_layers, vf_net_features = build_fc_layers(shared_net.features_dim, vf_layers, device=device)

        super().__init__(
            shared_net=shared_net,
            action_space=action_space,
            pi_net=th.nn.Sequential(*pi_layers),
            pi_net_features=pi_net_features,
            vf_net=th.nn.Sequential(*vf_layers),
            vf_net_features=vf_net_features,
            device=device
        )