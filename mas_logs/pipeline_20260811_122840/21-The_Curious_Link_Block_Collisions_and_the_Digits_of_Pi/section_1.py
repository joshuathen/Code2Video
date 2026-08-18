from manim import *
import numpy as np

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Can blocks colliding count the digits of Pi?",
            "Two blocks bounce on a frictionless surface.",
            "Small Block A sits near a wall.",
            "Larger Block B blocks the path.",
            "Total collisions reveal the digits of Pi."
        ]
        self.setup_layout("The Hook: The Bouncing Block Paradox", lecture_lines)
        
        # Assets
        wall = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg", color="#4D4D4D")
        block_a = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg", color=WHITE)
        block_b = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg", color=WHITE)
        
        # Positions
        self.place_at_grid(wall, 'B3', scale_factor=0.6)
        self.place_at_grid(block_a, 'C3', scale_factor=0.6)
        self.place_at_grid(block_b, 'C5', scale_factor=0.8)
        
        self.add(wall, block_a, block_b)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF5733"))
        self.play(block_a.animate.shift(RIGHT*0.5), block_b.animate.shift(LEFT*0.5), run_time=1)
        # Bounce simulation (change color to #FF5733 at peak)
        self.play(block_a.animate.set_color("#FF5733"), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#33FF57"))
        self.play(block_a.animate.set_color("#33FF57"), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#3357FF"))
        self.play(block_b.animate.set_color("#3357FF"), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FF33F6"))
        self.wait(2)
