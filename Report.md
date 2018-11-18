## Report

### Learning Algorithim

[Paper](https://arxiv.org/pdf/1509.02971.pdf)

![alt text](image_alg.png "DDPG Algorithim")

### Model Archetecture

I. *Initial model* archetecture used 3 layers for the Actor:

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))
        
and in the Critic:

        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        
        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        
II. *Second model* added batch normalization.  Actor:

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)
        
        x = F.relu(self.fc1(state))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))

Critic:
        
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.bn1 = nn.BatchNorm1d(fcs1_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)
        
        xs = F.relu(self.fcs1(state))
        xs = self.bn1(xs)
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
        
### Hyperparameters

        BUFFER_SIZE = int(1e5)  # replay buffer size
        BATCH_SIZE = 128        # minibatch size
        GAMMA = 0.99            # discount factor
        TAU = 1e-3              # for soft update of target parameters
        LR_ACTOR = 2e-4         # learning rate of the actor 
        LR_CRITIC = 2e-4        # learning rate of the critic
        WEIGHT_DECAY = 0        # L2 weight decay

### Results

Set maximum episode to 500.  

I.  Initial Model, fc1_units = 400, fc2_units = 300, fcs1_units=400, fc2_units=300.  Did not achieve an average score of 30:

![alt text](first_attempt.png "Result I")

II.  Initial Model, fc1_units = 128, fc2_units = 128, fcs1_units=128, fc2_units=128.  Did not acheive an average score of 30:

        Episode 100	Average Score: 2.67
        Episode 200	Average Score: 15.42
        Episode 300	Average Score: 22.28
        Episode 346	Average Score: 18.42

III.  Second Model, fc1_units = 128, fc2_units = 128, fcs1_units=128, fc2_units=128, with Batch Normalization.  Solved:
        
        Episode 100	Average Score: 4.56
        Episode 200	Average Score: 27.88
        Episode 219	Average Score: 30.10
        
![alt text](third_attempt.png "Result I")
        
