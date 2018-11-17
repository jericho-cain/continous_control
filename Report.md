## Report

### Learning Algorithim

### Model Archetecture

The archetecture used 3 layers:

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))
        
Where, fc1_units = 400, fc2_units = 300.



### Hyperparameters

### Results
