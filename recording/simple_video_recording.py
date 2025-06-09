from video_compatible_gym import VideoCompatibleVizDoomGym
import cv2
import numpy as np
import os

def evaluate_with_video_recording(model, n_eval_episodes=5, output_video="presentation_video.mp4", fps=15, scenario_type="deadly_corridor"):
    """
    Evaluate model and create a single compilation video with all episodes at slower speed
    """
    print(f"Recording {n_eval_episodes} episodes into a single video...")
    
    env = VideoCompatibleVizDoomGym(
        config="github/ViZDoom/scenarios/deadly_corridor_s1.cfg", 
        render_mode="rgb_array", 
        scenario_type=scenario_type
    )
    
    all_frames = []
    total_rewards = []
    
    for episode in range(n_eval_episodes):
        print(f"Recording episode {episode + 1}/{n_eval_episodes}")
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        episode_frames = []
        
        while not done:
            # Get the rendered frame
            frame = env.render(mode="rgb_array")
            if frame is not None:
                episode_frames.append(frame)
            
            # Get action from model and step
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1} finished. Reward: {episode_reward:.2f}")
        
        # Add episode title frame
        if episode_frames:
            # Create a title frame
            title_frame = np.zeros_like(episode_frames[0])
            cv2.putText(title_frame, f"Episode {episode + 1}", (50, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            cv2.putText(title_frame, f"Reward: {episode_reward:.1f}", (50, 140), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            
            # Add title frame for 2 seconds (2 * fps frames)
            for _ in range(2 * fps):
                all_frames.append(title_frame)
            
            # Add episode frames
            all_frames.extend(episode_frames)
            
            # Add separator frames (1 second pause)
            if episode < n_eval_episodes - 1:
                separator = np.zeros_like(episode_frames[0])
                for _ in range(fps):  # 1 second pause
                    all_frames.append(separator)
    
    env.close()
    
    # Save the compilation video
    if all_frames:
        height, width = all_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        
        for frame in all_frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"Compilation video saved as: {output_video}")
    
    mean_reward = np.mean(total_rewards)
    print(f"Mean reward: {mean_reward:.2f}")
    
    return mean_reward

# Usage in your notebook:
# from simple_video_recording import evaluate_with_video_recording
# mean_reward = evaluate_with_video_recording(model, n_eval_episodes=5, output_video="my_agent_demo.mp4") 