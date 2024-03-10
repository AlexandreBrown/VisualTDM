from envs.exploration import explore_env


if __name__ == "__main__":
    explore_env(env_name="AntMaze_UMaze", video_fps=30, seed=42)