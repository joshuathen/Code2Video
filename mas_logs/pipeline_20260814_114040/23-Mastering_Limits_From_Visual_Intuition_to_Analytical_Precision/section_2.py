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
        self.setup_layout("Formal Rigor: The Epsilon-Delta Definition", [
            "Formal limits require epsilon and delta.", 
            "Epsilon defines the allowable error range.", 
            "Delta defines the input distance constraint."
        ])
        
        # Elements for animation
        limit_point = 2
        limit_val = 5
        
        # Axis and line
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 10, 2], axis_config={"include_tip": False})
        line = axes.plot(lambda x: 2*x + 1, x_range=[0, 4], color="#00FFFF")
        
        # Applying requested placement fixes
        self.place_in_area(axes, 'B2', 'E5', scale_factor=0.9)
        self.place_in_area(line, 'B2', 'E5', scale_factor=0.9)
        
        # Epsilon-Delta setup (relative to axes)
        epsilon = 0.4
        delta = epsilon / 2
        
        # y_range (epsilon) and x_range (delta)
        # Using axes.c2p to ensure correct placement
        y_range = Rectangle(height=axes.c2p(0, limit_val+epsilon)[1] - axes.c2p(0, limit_val-epsilon)[1], width=0.5, color="#FFC0CB", fill_opacity=0.3)
        y_range.move_to(axes.c2p(limit_point, limit_val))
        
        x_range = Rectangle(height=0.5, width=axes.c2p(limit_point+delta, 0)[0] - axes.c2p(limit_point-delta, 0)[0], color="#BFFF00", fill_opacity=0.3)
        x_range.move_to(axes.c2p(limit_point, 0))

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        self.play(Create(axes), Create(line))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFC0CB"))
        self.place_at_grid(y_range, 'B2', scale_factor=0.5)
        self.play(Create(y_range))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#BFFF00"))
        self.place_at_grid(x_range, 'E5', scale_factor=0.5)
        self.play(Create(x_range))
        
        self.wait(2)
