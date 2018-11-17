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
        
### Hyperparameters

### Results

I.  fc1_units = 400 and fc2_units = 300.  However, it could not achieve an average score of 30:

![alt text](first_attempt.png "Result I")

II.  fc1_units = 128 and fc2_units = 128.
