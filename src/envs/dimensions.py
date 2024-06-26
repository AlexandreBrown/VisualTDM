def get_dim(self, key: str) -> int:
    if key == "pixels_latent" or key == "goal_latent" or key == "desired_goal":
        return self.goal_dim
    if key == "state":
        return self.state_dim
    if key == "action":
        return self.actions_dim
    if key == "planning_horizon":
        return self.tdm_planning_horizon_dim
    raise ValueError(f"Unknown dimension for key '{key}'!")
