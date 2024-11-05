from __future__ import annotations
import cv2
import numpy as np
import skimage.measure

from gymnasium import spaces
from minigrid.core.belief_grid import BeliefGrid
from minigrid.core.world_object import Goal, Lava
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.mission import MissionSpace


class ImageToGrid(MiniGridEnv):

    def __init__(
        self,
        env_img: np.ndarray,
        agent_start_pos: tuple[int, int] = (2,58), # Make sure the start position doesn't overlap with any obstacles
        agent_start_dir: int = 0,
        goal_pos: tuple[int, int] | None = None, 
        width: int = 80, 
        height: int = 60,
        max_steps: int = 1000, 
        **kwargs
    ):  
        self.env_img = env_img
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.goal_pos = goal_pos 

        super().__init__(
            mission_space = MissionSpace(mission_func=self._gen_mission),
            width = width, 
            height = height, 
            max_steps = max_steps,
            **kwargs
        )

    @staticmethod
    def _gen_mission():
        return "get to the goal safely"

    def _gen_grid(self, width: int, height: int) -> None:
        # Create an empty grid
        self.grid = BeliefGrid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, self.width, self.height)

        # Blue Channel Threshold to be consider water 
        threshold = 175

        # Gaussian Blur to deal with labels and smaller streams
        blur_kernal_size = (77, 77)

        img = self.env_img

        # Trim image to fit x_dim and y_dim and make greyscale based off blue channel 
        img = img[:(img.shape[0] // self.height) * self.height, :(img.shape[1] // self.width) * self.width, 2]

        # Apply Gaussain Blue
        img = cv2.GaussianBlur(img, blur_kernal_size, 0)

        # Make image black or white
        img = (img > threshold).astype(np.uint8) * 255

        # Min Pooling to downscale image
        img = skimage.measure.block_reduce(img, (int(img.shape[0] / self.height), int(img.shape[1] / self.width)), np.min)

        coordinates_of_land = np.argwhere(img == 255)
        list_land = list(map(tuple, coordinates_of_land))
        for tup in list_land: 
            # Ignore if it already on the border
            if tup[0] != 0 and tup[0] != height - 1 and tup[1] != 0 and tup[1] != width - 1:
                self.grid.set(tup[1], tup[0], Lava())

        # If an agent start position is not definied, we will look for an empty spot
        self.agent_pos = self.agent_start_pos 
        self.agent_dir = self.agent_start_dir

        if self.goal_pos is not None:
            self.put_obj(Goal(), self.goal_pos[0], self.goal_pos[1])
    
# Example Testing Function 
# Alter or remove this based on what you're planning on doing    
def main():
    env = ImageToGrid(cv2.imread("env_images/chesapeake.png"), width = 80, height = 60, render_mode="human")
    manual_control = ManualControl(env, seed=42)
    manual_control.start()
if __name__ == "__main__":
    main()

