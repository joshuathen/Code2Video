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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Zero-Probability Paradox", [
            "Single point probability is zero.",
            "Continuous variable width is zero.",
            "Probability exists only over ranges."
        ])
        
        # Create PDF curve
        axes = Axes(x_range=[0, 4], y_range=[0, 1], axis_config={"include_numbers": False}).scale(0.7)
        curve = axes.plot(lambda x: np.exp(-(x-2)**2), x_range=[0, 4], color=BLUE)
        
        # Apply layout fixes from Critic
        self.place_in_area(axes, 'C2', 'E5', scale_factor=0.8)
        self.place_in_area(curve, 'C2', 'E5', scale_factor=0.8)
        
        x_val = 2.0
        point = Dot(axes.c2p(x_val, 0), color=WHITE)
        point_on_curve = Dot(axes.c2p(x_val, np.exp(-(x_val-2)**2)), color=WHITE)
        
        # Placeholder asset loading (as requested by instructions)
        # Assuming SVG for icon, but the path ends in .svg
        asset_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        self.place_at_grid(asset_icon, 'A2', scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.play(Create(point), Create(point_on_curve), FadeIn(asset_icon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF00FF"))
        width_rect = Rectangle(height=axes.c2p(0, 0.1)[1] - axes.c2p(0, 0)[1], 
                              width=0.2, color="#FF00FF", fill_opacity=0.5)
        width_rect.move_to(axes.c2p(x_val, 0.05))
        self.play(Create(width_rect))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF0000"))
        prob_text = MathTex("P(X=x) = 0", color="#FF0000").scale(0.8)
        self.place_at_grid(prob_text, 'B2', scale_factor=0.9)
        self.play(Write(prob_text))
        self.wait(2)
