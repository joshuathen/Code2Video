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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "PDEs require context to solve uniquely.",
            "Initial conditions set the state at time zero.",
            "Boundary conditions define constraints at the edges."
        ]
        self.setup_layout("Boundary and Initial Conditions", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/mat.svg]
        mat = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/mat.svg")
        self.place_in_area(mat, 'B4', 'E6', scale_factor=0.6)
        
        b_label = Text("Boundary", color="#4A90E2", font_size=20)
        self.place_at_grid(b_label, 'B4') 
        b_label.next_to(mat, UP)
        
        heat = Circle(radius=0.5, color="#FF5733", fill_opacity=0.8)
        self.place_at_grid(heat, 'C5', scale_factor=0.5)
        
        initial_condition_label = Text("Initial Condition", color="#FF5733", font_size=18)
        self.place_at_grid(initial_condition_label, 'D5', scale_factor=0.7)
        initial_condition_label.next_to(heat, DOWN)
        
        self.play(FadeIn(mat), Write(b_label), FadeIn(heat), Write(initial_condition_label))
        self.lecture[0].set_color("#4A90E2")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Animate the heat spreading towards the boundary
        diffusion = heat.copy().animate.scale(3.0).set_opacity(0.3)
        self.play(diffusion, run_time=2)
        self.lecture[1].set_color("#FF5733")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Display the boundaries darkening
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/mat.svg]
        mat_dark = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/mat.svg")
        self.place_in_area(mat_dark, 'B4', 'E6', scale_factor=0.6)
        mat_dark.set_color(DARK_GRAY)
        
        self.play(ReplacementTransform(mat, mat_dark), run_time=1.5)
        self.lecture[2].set_color("#FFFFFF")
        self.wait(2)
