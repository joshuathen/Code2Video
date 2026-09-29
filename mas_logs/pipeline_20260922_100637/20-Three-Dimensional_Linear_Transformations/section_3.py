from manim import *
import numpy as np
import os

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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Visualizing Transformations", [
            "Scaling stretches space along axes.", 
            "Rotation turns space around an axis.", 
            "Shearing slants the grid space.",
            "Visual scaling effect.",
            "Rotation in action."
        ])
        
        # Initial State: Unit square and Character asset
        square = Square(side_length=1.5, color=WHITE).set_stroke(width=2)
        self.place_at_grid(square, 'C4', scale_factor=0.6)
        
        char_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/character.png"
        if os.path.exists(char_path):
            char = ImageMobject(char_path)
        else:
            char = Dot(color=BLUE)
        self.place_at_grid(char, 'C2', scale_factor=0.4)
        
        self.add(square, char)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF5733")
        self.play(square.animate.stretch(2, dim=0), run_time=1.5)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#33FF57")
        self.play(square.animate.rotate(PI/4), run_time=1.5)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#3357FF")
        self.play(square.animate.apply_matrix([[1, 0.5], [0, 1]]), run_time=1.5)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFFF33")
        scaling_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/digital_character_scaling.png"
        if os.path.exists(scaling_path):
            scaling_asset = ImageMobject(scaling_path)
        else:
            scaling_asset = Square(color=YELLOW, side_length=0.5)
        self.place_at_grid(scaling_asset, 'B5', scale_factor=0.7)
        self.add(scaling_asset)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FF33FF")
        rotation_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/digital_character_rotation.png"
        if os.path.exists(rotation_path):
            rotation_asset = ImageMobject(rotation_path)
        else:
            rotation_asset = Circle(color="#FF00FF", radius=0.3)
        self.place_at_grid(rotation_asset, 'E2', scale_factor=0.7)
        self.add(rotation_asset)
        self.wait(1)
