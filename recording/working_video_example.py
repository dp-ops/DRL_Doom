from simple_video_recording import evaluate_with_video_recording
from stable_baselines3 import PPO

# Load your trained model
model = PPO.load('c:/Users/dimos/PyProjecks/DRL_Doom/train/train_deadly_v2/best_model_810000')

# Record evaluation episodes and save as single video
print("Recording evaluation episodes...")
mean_reward = evaluate_with_video_recording(
    model, 
    n_eval_episodes=5, 
    output_video="agent_deadly_corridor.mp4", 
    fps=10, 
    scenario_type="deadly_corridor"
)

print(f"Evaluation completed!")
print(f"Mean reward: {mean_reward:.2f}")
print("Check 'agent_deadly_corridor.mp4' for your complete gameplay demonstration!")
print("The video shows your agent navigating the deadly corridor with titles and slower playback for better visibility.") 