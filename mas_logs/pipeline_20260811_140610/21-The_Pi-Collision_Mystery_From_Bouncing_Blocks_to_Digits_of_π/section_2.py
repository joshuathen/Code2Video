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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Pattern: Counting the Collisions", [
            "Collisions occur between the blocks and wall.",
            "The count depends on the mass ratio.",
            "Mass ratio of 100 yields 31 collisions."
        ])
        
        # Assets
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg]
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg]
        
        small_block = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg", color=BLUE)
        large_block = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg", color=RED)
        wall = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg", color=WHITE)
        
        # Layout
        self.place_at_grid(wall, "D6", scale_factor=0.5)
        self.place_at_grid(large_block, "D4", scale_factor=1.2)
        self.place_at_grid(small_block, "D2", scale_factor=0.6)
        
        collision_counter = Text("0", font_size=36, color=YELLOW)
        self.place_at_grid(collision_counter, "A1")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.play(small_block.animate.shift(RIGHT * 1.5), run_time=1)
        self.play(small_block.animate.shift(LEFT * 0.5), large_block.animate.shift(RIGHT * 0.2), run_time=0.5)
        self.play(large_block.animate.shift(RIGHT * 0.3), run_time=0.5)
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(RED)
        self.play(large_block.animate.scale(1.5), run_time=1)
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        for i in range(1, 32):
            val = Text(str(i), font_size=36, color=YELLOW)
            self.play(Transform(collision_counter, val), run_time=0.05)
