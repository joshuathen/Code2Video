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
        title = "The Ghostly Result: ζ(-1) = -1/12"
        lines = [
            "In the extended domain, we find shocking new values.",
            "The sum of all natural numbers maps to negative twelfth.",
            "This result appears as a dip in the function's landscape."
        ]
        self.setup_layout(title, lines)
        
        # === Animation for Lecture Line 1 ===
        # Color match: Line 1 -> #FFFFFF
        self.lecture[0].set_color("#FFFFFF")
        
        # Display the sum 1 + 2 + 3 + 4... (#FFFFFF)
        sum_tex = Text("1 + 2 + 3 + 4 + ...", color="#FFFFFF", font_size=36)
        # Fix Issue 47: adjust area and scale to prevent overlap
        self.place_in_area(sum_tex, 'B2', 'B6', scale_factor=0.9)
        
        self.play(Write(sum_tex))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Color match: Line 2 -> #00FFFF
        self.lecture[1].set_color("#00FFFF")
        
        # Hold a cyan 'Magic Lens' (#00FFFF) over the sum.
        lens_circle = Circle(radius=1.2, color="#00FFFF", stroke_width=6)
        lens_handle = Line(
            start=lens_circle.get_bottom(), 
            end=lens_circle.get_bottom() + DOWN * 0.8 + RIGHT * 0.5, 
            color="#00FFFF", 
            stroke_width=8
        )
        magic_lens = VGroup(lens_circle, lens_handle)
        
        # Position lens. Fix Issue 48: move to lower area and rescale
        self.place_in_area(magic_lens, 'D3', 'F6', scale_factor=0.8)
        
        self.play(Create(magic_lens))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Color match: Line 3 -> #FF0000
        self.lecture[2].set_color("#FF0000")
        
        # The lens reveals -1/12 (#FF0000) etched into a grid.
        # 1. Create a background grid within the lens area (#FFFFFF)
        grid_bg = VGroup()
        for x_off in np.arange(-1.0, 1.1, 0.4):
            grid_bg.add(Line([x_off, -1.0, 0], [x_off, 1.0, 0], stroke_width=1, stroke_opacity=0.3, color=WHITE))
        for y_off in np.arange(-1.0, 1.1, 0.4):
            grid_bg.add(Line([-1.0, y_off, 0], [1.0, y_off, 0], stroke_width=1, stroke_opacity=0.3, color=WHITE))
        
        # Position grid background relative to current lens position
        grid_bg.move_to(lens_circle.get_center())
        
        # 2. Result value - Using Text instead of MathTex
        result_tex = Text("-1/12", color="#FF0000", font_size=48)
        result_tex.move_to(lens_circle.get_center())
        
        # Reveal animation: Fade out sum, show grid and result inside lens
        self.play(
            sum_tex.animate.set_opacity(0.1),
            FadeIn(grid_bg),
            Write(result_tex),
            run_time=2
        )
        self.wait(2)
