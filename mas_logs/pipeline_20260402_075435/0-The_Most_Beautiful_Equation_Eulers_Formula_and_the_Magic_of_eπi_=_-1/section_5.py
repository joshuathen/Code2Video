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
        title_text = "The Journey of π: Reaching the Destination"
        lecture_lines = [
            "Now, let our rotation travel a distance of pi.",
            "Pi radians represents exactly half a trip around.",
            "We land precisely at negative one on the axis."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors for alignment with lecture lines
        line_colors = [YELLOW, BLUE, GREEN]

        # === Animation for Lecture Line 1 ===
        # Now, let our rotation travel a distance of pi.
        # Replacing MathTex with Text to avoid LaTeX dependency
        formula = Text("e^ix", color=line_colors[0])
        self.place_in_area(formula, "A3", "A4", scale_factor=1.5)
        
        # Load the unit circle asset
        unit_circle = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/unit.svg")
        # Position in a central block of the right side grid
        self.place_in_area(unit_circle, "C2", "F5", scale_factor=2.0)
        
        # Center of the circle for geometry
        circle_center = unit_circle.get_center()
        # Visual radius based on grid scale (C2 to F5 width is ~3.0, so radius ~1.5)
        radius = 1.5

        self.play(
            Write(formula),
            self.lecture[0].animate.set_color(line_colors[0]),
            run_time=1
        )
        self.wait(0.5)
        
        # Using Text with unicode character for pi
        formula_pi = Text("e^πi", color=line_colors[0])
        self.place_in_area(formula_pi, "A3", "A4", scale_factor=1.5)
        
        self.play(
            Transform(formula, formula_pi),
            FadeIn(unit_circle),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Pi radians represents exactly half a trip around.
        
        point = Dot(color=line_colors[1], radius=0.1)
        point.move_to(circle_center + RIGHT * radius)
        
        # Path for the point movement (top half of unit circle)
        arc = Arc(
            radius=radius,
            start_angle=0,
            angle=PI,
            arc_center=circle_center,
            color=line_colors[1]
        )
        
        # Label for pi distance
        pi_label = Text("π", color=line_colors[1])
        self.place_at_grid(pi_label, "B3", scale_factor=1.2)
        
        self.play(
            FadeIn(point),
            self.lecture[1].animate.set_color(line_colors[1]),
            run_time=0.5
        )
        
        self.play(
            MoveAlongPath(point, arc),
            Create(arc),
            Write(pi_label),
            run_time=3,
            rate_func=smooth
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # We land precisely at negative one on the axis.
        
        neg_one_val = Text("-1", color=WHITE, font_size=32)
        # Position label slightly outside the final point position
        neg_one_val.move_to(circle_center + LEFT * (radius + 0.6))
        
        self.play(
            self.lecture[2].animate.set_color(line_colors[2]),
            Flash(point, color=WHITE, line_length=0.4),
            FadeIn(neg_one_val),
            run_time=1.5
        )
        
        # Final formula transformation to show the result
        formula_final = Text("e^πi = -1", color=line_colors[2])
        self.place_in_area(formula_final, "A3", "A4", scale_factor=1.5)
        
        self.play(
            Transform(formula, formula_final),
            point.animate.scale(1.5),
            run_time=1.5
        )

        self.wait(3)
