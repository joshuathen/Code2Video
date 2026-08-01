from manim import *
import numpy as np
from pathlib import Path

# Fix for potential KeyError or formatting issues with path strings
config.input_file = Path("section_5.py")

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
        title = "The Final Calculation: Plugging in π"
        lines = [
            "Let's set the angle theta to exactly pi.", 
            "In radians, pi represents a half-circle rotation.", 
            "Starting at one, we rotate one hundred eighty degrees.", 
            "We land exactly at negative one on the axis.", 
            "Therefore, e to the i*pi equals negative one."
        ]
        self.setup_layout(title, lines)
        
        # Colors for highlights
        h_colors = ["#FFFF00", "#50C878", "#1F51FF", "#FF3131", "#E0B0FF"]

        # === Animation for Lecture Line 1 ===
        # Show prominent e^iπ
        self.lecture[0].set_color(h_colors[0])
        formula_pi = Text("e^iπ", font_size=48, color=h_colors[0])
        # [Issue 30 Fix]: Moving from A3 to A4
        self.place_at_grid(formula_pi, "A4", scale_factor=1.2)
        self.play(Write(formula_pi))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Setup complex plane and show pi rotation concept
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(h_colors[1])
        
        # [Issue 31 Fix]: Moving plane area from C2-E5 to C3-E6
        plane = ComplexPlane(
            x_range=[-1.5, 1.5, 1],
            y_range=[-1.5, 1.5, 1],
            background_line_style={"stroke_opacity": 0.3}
        )
        self.place_in_area(plane, "C3", "E6", scale_factor=1.2)
        
        unit_circle = Circle(
            radius=plane.get_x_unit_size(), 
            color=WHITE, 
            stroke_opacity=0.3
        ).move_to(plane.get_origin())
        
        # Label "pi radians" in #FFFF00 as per prompt description
        pi_label = Text("π radians", font_size=24, color="#FFFF00")
        self.place_at_grid(pi_label, "B4", scale_factor=1.0)

        self.play(Create(plane), Create(unit_circle))
        self.play(FadeIn(pi_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Rotation from 1 to -1
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(h_colors[2])
        
        dot = Dot(plane.n2p(1), color=h_colors[2])
        tracing_arc = Arc(
            radius=plane.get_x_unit_size(),
            start_angle=0,
            angle=PI,
            color=h_colors[2]
        ).move_to(plane.get_origin())
        
        self.play(FadeIn(dot))
        self.play(
            MoveAlongPath(dot, tracing_arc), 
            Create(tracing_arc), 
            run_time=2, 
            rate_func=linear
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Landing and labeling -1
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(h_colors[3])
        
        landing_pos = plane.n2p(-1)
        target_marker = Cross(scale_factor=0.2, stroke_width=4, color=h_colors[3]).move_to(landing_pos)
        
        # Using grid-relative positioning for the label to stay consistent with constraints
        label_neg_1 = Text("-1", font_size=36, color=h_colors[3])
        self.place_at_grid(label_neg_1, "E3", scale_factor=0.8) # Adjusted to be near the point in C3-E6 area
        
        self.play(Create(target_marker), Write(label_neg_1))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Final identity display and rearrangement
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(h_colors[4])
        
        final_identity = Text("e^iπ = -1", font_size=48, color=h_colors[4])
        # [Issue 32 Fix]: Moving from F3 to F4
        self.place_at_grid(final_identity, "F4", scale_factor=1.0)
        
        self.play(Write(final_identity))
        self.wait(1)
        
        # Rearranging to e^{iπ} + 1 = 0 as requested in description
        rearranged_identity = Text("e^iπ + 1 = 0", font_size=48, color=h_colors[4])
        self.place_at_grid(rearranged_identity, "F4", scale_factor=1.0)
        
        self.play(Transform(final_identity, rearranged_identity))
        self.play(Indicate(final_identity))
        self.wait(3)
