from manim import *
import numpy as np
from pathlib import Path

# Override the input_file configuration to a safe path string.
config.input_file = Path("section_1.py")

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
            'Five major mathematical constants appear from different worlds.',
            'They converge into a single, unified cluster.',
            'Euler’s bridge connects these fundamental numbers together.'
        ]
        self.setup_layout("The Grand Reunion", lecture_lines)

        # Define colors for the mathematical constants
        color_e = "#00FF00"
        color_i = "#00FFFF"
        color_pi = "#FF8800"
        color_1 = "#FFFF00"
        color_0 = "#FFFFFF"

        # Create constants using Text and Unicode symbols (No TeX/MathTex)
        const_e = Text("e", color=color_e, font_size=60)
        const_i = Text("i", color=color_i, font_size=60)
        const_pi = Text("π", color=color_pi, font_size=60)
        const_1 = Text("1", color=color_1, font_size=60)
        const_0 = Text("0", color=color_0, font_size=60)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(color_e))
        
        # Fade in Title Text "Mathematics' Greatest Constants"
        intro_text = Text("Mathematics' Greatest Constants", font_size=32, color=WHITE)
        self.place_in_area(intro_text, 'B2', 'D5', scale_factor=0.7)
        self.play(FadeIn(intro_text))
        self.wait(1)
        self.play(FadeOut(intro_text))

        # Fade in symbols 0, 1, pi, e, and i at center C3
        constants = [const_e, const_i, const_pi, const_1, const_0]
        # Set initial scale for all symbols per Issue 51
        for c in constants:
            c.scale(0.8)
            c.move_to(self.grid['C3'])
        
        self.play(FadeIn(VGroup(*constants)), run_time=1.5)
        self.wait(0.5)

        # Scatter the symbols to different corners (separation)
        # Positions based on Issue 51: e->B2, i->B4, pi->D2, 1->D4, 0 stays at center C3
        self.play(
            const_e.animate.move_to(self.grid['B2']),
            const_i.animate.move_to(self.grid['B4']),
            const_pi.animate.move_to(self.grid['D2']),
            const_1.animate.move_to(self.grid['D4']),
            # const_0 stays at C3
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(color_1))
        
        # Move symbols to the center cluster (near C3)
        center_pos = self.grid['C3']
        self.play(
            const_e.animate.move_to(center_pos + UP * 0.7),
            const_i.animate.move_to(center_pos + RIGHT * 0.7),
            const_pi.animate.move_to(center_pos + DOWN * 0.7),
            const_1.animate.move_to(center_pos + LEFT * 0.7),
            const_0.animate.move_to(center_pos),
            run_time=2
        )
        
        # Connect with glowing white lines (#FFFFFF)
        lines = VGroup(
            Line(const_e.get_center(), const_0.get_center(), color=WHITE, stroke_width=2),
            Line(const_i.get_center(), const_0.get_center(), color=WHITE, stroke_width=2),
            Line(const_pi.get_center(), const_0.get_center(), color=WHITE, stroke_width=2),
            Line(const_1.get_center(), const_0.get_center(), color=WHITE, stroke_width=2),
        )
        # Create a subtle glow effect with high-width low-opacity duplicates
        glows = lines.copy().set_stroke(width=6, opacity=0.2)
        
        self.play(Create(lines), FadeIn(glows), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(color_pi))
        
        # Fade in a light gray coordinate grid (#555555) behind the symbols
        grid_lines = VGroup()
        for i in range(1, 7): # Column vertical lines
            grid_lines.add(Line(self.grid[f"A{i}"], self.grid[f"F{i}"], color="#555555", stroke_width=1))
        for r in ["A", "B", "C", "D", "E", "F"]: # Row horizontal lines
            grid_lines.add(Line(self.grid[f"{r}1"], self.grid[f"{r}6"], color="#555555", stroke_width=1))
        
        # Ensure grid is rendered behind objects
        grid_lines.set_z_index(-1)
        self.play(FadeIn(grid_lines, shift=IN), run_time=1.5)
        self.wait(2)
