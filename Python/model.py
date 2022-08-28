import torch
from torch import nn
import torch.nn.functional as F
from collections import namedtuple
import random
import torchvision
import torchvision.transforms as T

class DQN(nn.Module):
    def __init__(self, imsize, in_channels, out_channels, batch_norm=True):
        super(DQN, self).__init__()
        
        # Convulutional input layers
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        
        # Batch normailization
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn1 = nn.BatchNorm2d(16)
            self.bn2 = nn.BatchNorm2d(32)
            self.bn3 = nn.BatchNorm2d(32)
        

        # Fully connected output layer
        imsize = imsize if type(imsize) is tuple else (imsize,imsize)
        n_hidden = ((imsize[0]-21)//8)*((imsize[1]-21)//8) * 32
        self.head = nn.Linear(n_hidden, out_channels)

    def forward(self, x):
        if self.batch_norm:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
        else:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
        return self.head(x.flatten(start_dim=1))


class AlexNet(nn.Module):
    def __init__(self, imsize, in_channels, out_channels,
                 normalize_input=True, n_trainable_first_layers=None, n_trainable_last_layers=None):
        super(AlexNet, self).__init__()

        # Load pretrained model
        self.model = torchvision.models.alexnet(pretrained=True)

        # Freeze all parameters except those of the last layers / the first layers
	# (Note that the very first and the very last layers are replaced, and thus set trainable)
        for param in self.model.parameters():
            param.requires_grad = False

        if n_trainable_first_layers is not None:
            convlayers = [layer for layer in self.model.features if 'Conv' in layer._get_name()]
            for layer in convlayers[:n_trainable_first_layers]:
                for param in layer.parameters():
                    param.requires_grad = True

        if n_trainable_last_layers is not None:
            fclayers = [layer for layer in self.model.features if 'Linear' in layer._get_name()]
            for layer in fclayers[::-1][:n_trainable_last_layers]:
                for param in layer.parameters():
                    param.requires_grad = True

        # Replace first and last layer of model
        self.model.features[0] = nn.Conv2d(in_channels,64,kernel_size=(11,11),stride=(4,4),padding=(2,2))
        self.model.classifier[-1] = nn.Linear(in_features=4096, out_features=out_channels,bias=True)

        # Normalization
        self.normalize_input = normalize_input
        if normalize_input:
            self.normalizer = lambda x, mean=0.445, std=0.269: (x - mean)/std # assuming (frame-stacked) greyscale images

        # Other settings (not used here)
        self.imsize = imsize
        self.in_channels = in_channels


    def forward(self, x):
        assert len(x.shape) == 4 # B, C, H, W

        if self.normalize_input:
            x = self.normalizer(x)

        if x.shape[-1]< 244:
            x = nn.functional.interpolate(x,size=244)

        if x.shape[1] == 1:
            x = x.repeat(1,3,1,1)

        return self.model(x).flatten(start_dim=1)


#replay memory
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DoubleDQNAgent():
    def __init__(self,
                 imsize=128,
                 in_channels=None,
                 n_actions=3,
                 memory_capacity=12000,
                 eps_start=0.95,
                 eps_end=0.05,
                 eps_delta=(0.05-0.95)/4000,
                 gamma_discount = 0.999,
                 batch_size = 128,
                 batch_norm = True,
                 device='cpu',
                 pretrained_model = None,
                 *args,**kwargs):

        # DQNs
        self.imsize      = imsize
        self.n_actions   = n_actions
        self.device      = torch.device(device)

        # Input channels (by default equal to stack size)
        if in_channels is not None:
            self.in_channels = int(in_channels)
        else:
            self.in_channels = int(kwargs['stack_size'])

        if pretrained_model is None:
            self.policy_net = DQN(imsize, self.in_channels, n_actions, batch_norm).to(device)
            self.target_net = DQN(imsize, self.in_channels, n_actions, batch_norm).to(device)
        else:
            print('initializing with pretrained model: {}'.format(pretrained_model))
            if pretrained_model == 'AlexNet':
                self.policy_net = AlexNet(imsize, self.in_channels, n_actions).to(device)
                self.target_net = AlexNet(imsize, self.in_channels, n_actions).to(device)
            else:
                raise NotImplementedError

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Replay memory
        self.memory = ReplayMemory(memory_capacity)

        # Epsilon decay
        self.eps_start  = eps_start
        self.eps_end    = eps_end
        self.eps_delta  = eps_delta
        self.step_count = 0

        # Gamma discount for future rewards
        self.gamma_discount = gamma_discount
        self.batch_size     = batch_size

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def select_action(self,state, validation=False):
        if not validation:
            sample = torch.rand(1)
            self.step_count += 1
        else:
            sample = 1
        self.eps_threshold = max(self.eps_end, (self.eps_start+(self.eps_delta*self.step_count)))
        if sample > self.eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).argmax(1)
        else:
            return torch.randint(self.n_actions,(1,),device=self.device)

    def forward(self):
        """implementation from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html"""
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch).squeeze()

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma_discount) + reward_batch

        return state_action_values, expected_state_action_values
