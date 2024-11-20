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
        self.grid = BeliefGrid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, self.width, self.height)

        # Blue Channel Threshold to be consider water 
        blue_color = np.array([253, 183, 147])

        img = self.env_img

        # Trim image to fit make it a multiple of x_dim and y_dim 
        img = img[:(img.shape[0] // self.height) * self.height, :(img.shape[1] // self.width) * self.width]

        # Discrimate pixels of our desired color
        img = (img[:, :, 0] == blue_color[0]) & (img[:, :, 1] == blue_color[1]) & (img[:, :, 2] == blue_color[2])
        img = (img).astype(np.uint8)
        img = img * 255

        # Min Pooling to downscale image
        img = skimage.measure.block_reduce(img, (int(img.shape[0] / self.height), int(img.shape[1] / self.width)), np.min)

        coordinates_of_land = np.argwhere(img == 0)
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
    env = ImageToGrid(cv2.imread("env_images/st_thomas_no_labels.png"), width = 80, height = 60, render_mode="human")
    manual_control = ManualControl(env, seed=42)
    manual_control.start()
if __name__ == "__main__":
    main()