from utils.DoomGym import VizDoomGym, VizDoomGym_DeadlyCorridor
import gymnasium as gym
import numpy as np
import cv2

class VideoCompatibleVizDoomGym(gym.Wrapper):
    """
    Wrapper for VizDoomGym that makes it compatible with video recording
    by properly implementing the render method to return rgb_array
    """
    
    def __init__(self, config="github/ViZDoom/scenarios/deadly_corridor_s1.cfg", render_mode="rgb_array", scenario_type="deadly_corridor"):
        # Create the base environment with rendering enabled
        if scenario_type == "deadly_corridor":
            base_env = VizDoomGym_DeadlyCorridor(render=True, config=config)
        else:
            base_env = VizDoomGym(render=True, config=config)
        super().__init__(base_env)
        
        # Set the render mode as a private attribute
        self._render_mode = render_mode
        
        # Store metadata for video recording
        self.metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 35}
    
    @property
    def render_mode(self):
        return self._render_mode
    
    def render(self, mode=None):
        """
        Render the environment.
        
        Args:
            mode: The render mode. Can be 'human' or 'rgb_array'
            
        Returns:
            If mode is 'rgb_array', returns the screen buffer as RGB array
            If mode is 'human', displays the window (default VizDoom behavior)
        """
        if mode is None:
            mode = self._render_mode
            
        if mode == "rgb_array":
            # Get the current state
            state = self.env.game.get_state()
            if state is not None:
                # Get screen buffer and convert to RGB format
                screen_buffer = state.screen_buffer
                # Convert from (channels, height, width) to (height, width, channels)
                frame = np.transpose(screen_buffer, (1, 2, 0))
                # Convert from BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return frame_rgb
            else:
                # Return a black frame if no state available
                return np.zeros((240, 320, 3), dtype=np.uint8)
                
        elif mode == "human":
            # Default VizDoom rendering (window display)
            # The window should already be visible since we created env with render=True
            pass
        
        return None 