class TaxiOption:
    def __init__(self, target_pos, env):
        self.target_pos = target_pos
        self.env = env
        
    def get_action(self, state):
        taxi_row, taxi_col, _, _ = self.env.unwrapped.decode(state)
        t_row, t_col = self.target_pos
        
        if taxi_row > t_row:
            return 1  # North
        elif taxi_row < t_row:
            return 0  # South
        if taxi_col > t_col:
            return 3  # West
        else:
            return 2  # East
    
    def is_terminated(self, state):
        taxi_row, taxi_col, _, _ = self.env.unwrapped.decode(state)
        return (taxi_row, taxi_col) == self.target_pos
