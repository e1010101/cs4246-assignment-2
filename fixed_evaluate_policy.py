def evaluate_policy(policy_net, env, num_eval_episodes=10, render=False, render_path=None):
    policy_net.eval()  # Set the network to evaluation mode
    total_rewards = []
    env_features = list(env.observation_space.keys())

    # Initialize visualizer if rendering
    if render and MovieGenerator is not None and render_path is not None:
        # Need to re-initialize env with rendering enabled for evaluation
        eval_env = Elevator(instance=5, is_render=True, render_path=render_path)
        print(f"Rendering evaluation to {render_path}")
    else:
        eval_env = env  # Use the original env if not rendering
        render = False  # Ensure render is False if MovieGenerator failed

    print(f"\nStarting evaluation for {num_eval_episodes} episodes...")
    for episode in range(num_eval_episodes):
        state = eval_env.reset()
        episode_reward = 0

        for t in range(eval_env.horizon):
            state_desc = eval_env.disc2state(state)
            state_list = convert_state_to_list(state_desc, env_features)
            state_tensor = torch.tensor(state_list, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                q_values = policy_net(state_tensor)
                action = q_values.max(1)[1].item()  # Choose action with highest Q-value

            next_state, reward, done, _ = eval_env.step(action)

            if render:
                eval_env.render()

            episode_reward += reward
            state = next_state

            if done:
                break

        total_rewards.append(episode_reward)
        print(f"Evaluation Episode {episode + 1}/{num_eval_episodes}, Reward: {episode_reward:.2f}")

    if render:
        try:
            eval_env.save_render()
            gif_path = os.path.join(render_path, 'elevator_eval.gif')
            print(f"Evaluation GIF saved to {gif_path}")
            
            # Check if the file exists before trying to display it
            if os.path.exists(gif_path):
                try:
                    display(Image(filename=gif_path))
                except NameError:
                    print("Cannot display image directly. Check the saved file.")
            else:
                print(f"Warning: GIF file was not created at {gif_path}. This may be due to missing dependencies like cv2.")
                print("You can still see the evaluation results in the console output.")
        except Exception as e:
            print(f"Error during rendering: {e}")
            print("Continuing with evaluation results...")
        
        eval_env.close()

    mean_eval_reward = np.mean(total_rewards)
    std_eval_reward = np.std(total_rewards)
    print(f"\nEvaluation Results ({num_eval_episodes} episodes):")
    print(f"Mean Reward: {mean_eval_reward:.2f} +/- {std_eval_reward:.2f}")
    policy_net.train()  # Set the network back to training mode
    return mean_eval_reward
