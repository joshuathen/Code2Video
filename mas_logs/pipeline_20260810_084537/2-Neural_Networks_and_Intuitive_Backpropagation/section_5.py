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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Gradient Descent: Refining the Model", [
            "Gradient descent refines the model's weights.",
            "Think of a hiker descending in fog.",
            "Small steps avoid overshooting the lowest point."
        ])
        
        # --- Prepare Assets ---
        axes = Axes(x_range=[-2, 2, 1], y_range=[0, 1, 0.5], axis_config={"include_tip": False}, x_length=4, y_length=2)
        surface = axes.plot(lambda x: 0.25 * x**2, x_range=[-2, 2], color="#8A2BE2")
        
        # Use assets as required
        fog = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/fog.svg")
        hiker = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hiker.svg")
        
        bowl = VGroup(axes, surface)
        # Apply visual adjustments from feedback (Issue 33 / 38)
        self.place_in_area(bowl, 'B4', 'D6', scale_factor=0.55)
        
        # Position fog (Issue 32 / 38)
        self.place_in_area(fog, 'E3', 'F5', scale_factor=0.4)
        
        # === Animation for Lecture Line 1 ===
        self.play(Create(axes), Create(surface), FadeIn(fog))
        self.lecture[0].set_color("#8A2BE2")

        # === Animation for Lecture Line 2 ===
        # Position hiker at start point on the curve
        hiker.scale(0.5)
        start_pos = axes.c2p(1.5, 0.25 * 1.5**2)
        hiker.move_to(start_pos)
        
        self.play(FadeIn(hiker))
        self.lecture[1].set_color("#FFFFFF")

        # === Animation for Lecture Line 3 ===
        # Path for hiker toward minimum
        end_pos = axes.c2p(0, 0)
        self.play(hiker.animate.move_to(end_pos), run_time=2.0)
        
        min_indicator = Dot(color="#FFD700").scale(0.8)
        min_indicator.move_to(end_pos)
        self.play(FadeIn(min_indicator))
        self.lecture[2].set_color("#FFD700")
        
        self.wait(2)
