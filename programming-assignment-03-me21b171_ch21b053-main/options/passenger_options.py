class PassengerOption:
    def __init__(self, env):
        self.env = env
        
    def get_action(self, state):
        taxi_row, taxi_col, pass_loc, dest_idx = self.env.unwrapped.decode(state)
        
        # Determine target based on passenger location
        if pass_loc == 4:  # Passenger in taxi
            target = self.env.unwrapped.locs[dest_idx]
        else:
            target = self.env.unwrapped.locs[pass_loc]
            
        # Greedy navigation
        if taxi_row > target[0]:
            return 1  # North
        elif taxi_row < target[0]:
            return 0  # South
        if taxi_col > target[1]:
            return 3  # West
        else:
            return 2  # East
    
    def is_terminated(self, state):
        _, _, pass_loc, dest_idx = self.env.unwrapped.decode(state)
        return pass_loc == dest_idx  # Passenger at destination
