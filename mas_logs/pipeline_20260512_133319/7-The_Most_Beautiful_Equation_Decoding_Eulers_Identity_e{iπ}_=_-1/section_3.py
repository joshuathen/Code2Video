from manim import *
import numpy as np

# Fix: Manim CE v0.19.0 has an internal bug where braces in the file path (e.g., '{iπ}') 
# cause a KeyError during path formatting. We override the input_file to a safe name 
# to bypass this internal error during initialization.
config.input_file = "section_3.py"

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
        # Initial Setup
        title = "Prerequisite: 'e' and the Speed of Growth"
        lines = [
            "Constant e represents the limit of continuous growth.",
            "In the real world, it scales quantities exponentially.",
            "But direction matters just as much as speed."
        ]
        self.setup_layout(title, lines)
        
        # Colors
        GREEN_COL = "#00FF00"
        CYAN_COL = "#00FFFF"
        YELLOW_COL = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        # Show 'e' in center labeled 'Base of Growth'
        # Issue 52/35 Fix: e_symbol -> Grid C4, scale 1.1
        e_symbol = Text("e", font_size=80, color=GREEN_COL, slant="ITALIC")
        e_label = Text("Base of Growth", font_size=24, color=GREEN_COL)
        
        self.place_at_grid(e_symbol, "C4", scale_factor=1.1)
        e_label.next_to(e_symbol, DOWN, buff=0.2)
        
        self.play(self.lecture[0].animate.set_color(GREEN_COL))
        self.play(Write(e_symbol), FadeIn(e_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Animate a point moving right on an axis exponentially.
        # Issue 52/36 Fix: growth_dot -> Grid E1, scale 1.2
        axis_line = Line(start=self.grid["E1"], end=self.grid["E6"], color=GREY)
        ticks = VGroup(*[
            Line(UP*0.1, DOWN*0.1).move_to(self.grid[f"E{i}"]) 
            for i in range(1, 7)
        ])
        axis = VGroup(axis_line, ticks)
        
        growth_dot = Dot(color=CYAN_COL)
        self.place_at_grid(growth_dot, "E1", scale_factor=1.2)
        
        self.play(self.lecture[1].animate.set_color(CYAN_COL))
        self.play(Create(axis))
        self.play(FadeIn(growth_dot))
        
        # Exponential movement simulation: accelerating across the grid points
        self.play(growth_dot.animate.move_to(self.grid["E2"]), run_time=0.8, rate_func=linear)
        self.play(growth_dot.animate.move_to(self.grid["E4"]), run_time=0.6, rate_func=linear)
        self.play(growth_dot.animate.move_to(self.grid["E6"]), run_time=0.4, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight 'x' in the formula e^x
        # Issue 52/34 Fix: formula_group -> Grid B4, scale 0.8
        base_e_text = Text("e", font_size=60, slant="ITALIC")
        exp_x_text = Text("x", font_size=40, color=YELLOW_COL)
        
        # Create a formula group using VGroup
        formula_group = VGroup(base_e_text, exp_x_text)
        self.place_at_grid(formula_group, "B4", scale_factor=0.8)
        # Manually position 'x' as an exponent relative to the placed 'e'
        exp_x_text.move_to(base_e_text.get_corner(UR) + RIGHT*0.1 + UP*0.1)
        
        self.play(self.lecture[2].animate.set_color(YELLOW_COL))
        self.play(Write(base_e_text))
        self.play(Write(exp_x_text))
        
        # Pulse highlight on x exponent
        self.play(exp_x_text.animate.scale(1.3), run_time=0.5)
        self.play(exp_x_text.animate.scale(1/1.3), run_time=0.5)
        
        self.wait(3)
