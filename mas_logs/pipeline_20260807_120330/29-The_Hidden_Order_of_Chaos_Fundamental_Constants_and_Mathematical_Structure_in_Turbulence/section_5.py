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
        # Setup layout
        self.setup_layout("The Final Stop: Kolmogorov Microscales", [
            "Swirls eventually reach the Kolmogorov microscale.",
            "Here, viscosity transforms kinetic energy into heat.",
            "This dissipation stage ends the energy cascade."
        ])
        
        # Color definitions
        GREEN_COLOR = "#00FF00"
        RED_COLOR = "#FF0000"
        
        # === Animation for Lecture Line 1 ===
        # L: 'Swirls eventually reach the Kolmogorov microscale.'
        # A: Scale down tiny green circles until they are barely visible points.
        self.play(self.lecture[0].animate.set_color(GREEN_COLOR))
        
        # Create tiny green dots representing the final eddies in the cascade
        # Placing them in the visual area (Cols 4-6) to satisfy [B021]
        dot_positions = ["B4", "B6", "C5", "D4", "D6", "E5", "C4", "E6"]
        dots = VGroup(*[Dot(radius=0.1, color=GREEN_COLOR) for _ in dot_positions])
        
        for i, pos in enumerate(dot_positions):
            self.place_at_grid(dots[i], pos)
            
        self.play(FadeIn(dots, lag_ratio=0.1))
        self.wait(0.5)
        
        # Scale down to points (viscosity begins to dominate)
        self.play(dots.animate.scale(0.3), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # L: 'Here, viscosity transforms kinetic energy into heat.'
        # A: Change the color of the points from green (#00FF00) to glowing red (#FF0000).
        self.play(self.lecture[1].animate.set_color(RED_COLOR))
        
        # Create glowing red glows to accompany the color change
        glows = VGroup(*[
            Dot(radius=0.2, color=RED_COLOR, fill_opacity=0.4) 
            for _ in range(len(dots))
        ])
        for glow, dot in zip(glows, dots):
            glow.move_to(dot.get_center())
            
        self.play(
            dots.animate.set_color(RED_COLOR),
            FadeIn(glows, lag_ratio=0.1),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # L: 'This dissipation stage ends the energy cascade.'
        # A: Fade the points into a soft red cloud representing heat dissipation.
        self.play(self.lecture[2].animate.set_color(RED_COLOR))
        
        # Soft red cloud representing heat dissipation
        # Fixed per Issue #36: self.place_in_area(heat_cloud, 'B4', 'F6', scale_factor=0.6)
        heat_cloud = Dot(radius=2.5, color=RED_COLOR, fill_opacity=0.25)
        self.place_in_area(heat_cloud, "B4", "F6", scale_factor=0.6)
        
        self.play(
            FadeOut(dots),
            FadeOut(glows),
            FadeIn(heat_cloud),
            heat_cloud.animate.scale(1.2),
            run_time=2
        )
        self.wait(2)
