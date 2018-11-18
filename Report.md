## Report

### Learning Algorithim

### Model Archetecture

Initial model archetecture used 3 layers:

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))
        
Second model added batch normalization:

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)
        
### Hyperparameters

### Results

I.  Initial Model, fc1_units = 400 and fc2_units = 300.  Did not achieve an average score of 30:

![alt text](first_attempt.png "Result I")

II.  Initial Model, fc1_units = 128 and fc2_units = 128.  Did not acheive an average score of 30:

        Episode 100	Average Score: 2.67
        Episode 200	Average Score: 15.42
        Episode 300	Average Score: 22.28
        Episode 346	Average Score: 18.42

III.  Second Model, fc1_units = 128 and fc2_units = 128, with Batch Normalization:
