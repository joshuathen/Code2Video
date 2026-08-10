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
            "Kernels are small matrices extracting features.",
            "Blur kernels average out neighboring pixels.",
            "Edge kernels highlight high-contrast boundaries."
        ]
        self.setup_layout("Prerequisite Visuals: The Kernel's Identity", lecture_lines)
        
        # Identity Kernel 3x3
        kernel_data = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        kernel = Matrix(kernel_data, h_buff=0.8, v_buff=0.8)
        pixel_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pixel.svg")
        
        # Requirement: Apply VideoCritic feedback
        # Line 0: self.place_at_grid(kernel, 'B4', scale_factor=1.2)
        self.place_at_grid(kernel, 'B4', scale_factor=1.2)
        
        # === Animation for Lecture Line 1 ===
        # Using asset pixel.svg as a decoration or part of the kernel logic
        self.play(Create(kernel), self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Sliding window (camera.svg)
        window = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg", color="#00FFFF")
        # Line 3: self.place_at_grid(highlight_frame, 'B4', scale_factor=1.2)
        # Using window as the highlight_frame
        self.place_at_grid(window, 'B4', scale_factor=1.2)
        
        self.play(FadeIn(window), self.lecture[1].animate.set_color("#00FFFF"))
        self.play(window.animate.shift(UP*0.2 + LEFT*0.2), run_time=0.5)
        self.play(window.animate.shift(DOWN*0.4), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line 9: self.place_in_area(grid_visual, 'A4', 'F6', scale_factor=0.9)
        # Grid visual placeholder
        grid_visual = Rectangle(width=2.5, height=2.5, color="#00FF00")
        self.place_in_area(grid_visual, 'A4', 'F6', scale_factor=0.9)
        
        self.play(self.lecture[2].animate.set_color("#00FF00"), Create(grid_visual))
        self.play(kernel.animate.set_color("#00FF00"), run_time=1)
        self.wait(1)
