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
        # Setup the layout with title and lecture lines
        # Mandatory structure with 5 lines
        self.setup_layout("The Half-Circle Journey (e^i\u03c0)", [
            "Let's set our angle x to exactly pi radians.",
            "Pi represents a journey halfway around the circle's edge.",
            "Starting at one, we rotate one hundred eighty degrees.",
            "Our path leads us directly to the value negative one.",
            "Thus, e to the i pi equals negative one."
        ])

        # === Animation for Lecture Line 1 ===
        # Line: "Let's set our angle x to exactly pi radians."
        self.play(self.lecture[0].animate.set_color("#FFA500"))
        # Using Text to avoid LaTeX dependency as per section guidelines
        formula_sub = Text("e^(i\u03c0)", font_size=32, color="#FFA500")
        self.place_at_grid(formula_sub, "A3", scale_factor=1.2)
        self.play(Write(formula_sub))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line: "Pi represents a journey halfway around the circle's edge."
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        plane = NumberPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            x_length=4,
            y_length=4,
            axis_config={"include_tip": True},
            background_line_style={"stroke_opacity": 0.3}
        )
        # Fix for Issue 40: Position plane in area B2-E5 to avoid bottom cutoff
        self.place_in_area(plane, 'B2', 'E5', scale_factor=1.0)
        
        circle = Circle(radius=plane.get_x_unit_size(), color=BLUE_B)
        circle.move_to(plane.get_origin())
        
        self.play(Create(plane), Create(circle))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line: "Starting at one, we rotate one hundred eighty degrees."
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        
        # Start point at '1'
        dot = Dot(plane.coords_to_point(1, 0), color="#00FF00")
        # Fix for Issue 41: Label '1' at grid D5
        label_one = Text("1", font_size=24, color=WHITE)
        self.place_at_grid(label_one, 'D5', scale_factor=0.7)
        
        # Arc path from 1 to -1 (pi radians)
        arc_path = Arc(
            radius=plane.get_x_unit_size(), 
            start_angle=0, 
            angle=PI, 
            color=YELLOW
        )
        arc_path.move_to(plane.get_origin(), aligned_edge=LEFT) # Standard arc 0-pi starts at its center's right
        arc_path.shift(LEFT * plane.get_x_unit_size()) # This is getting complex, let's just use shift.
        arc_path = Arc(radius=plane.get_x_unit_size(), start_angle=0, angle=PI, color=YELLOW)
        arc_path.shift(plane.get_origin())
        
        self.play(FadeIn(dot), Write(label_one))
        self.play(
            MoveAlongPath(dot, arc_path),
            Create(arc_path),
            run_time=3,
            rate_func=slow_into
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Line: "Our path leads us directly to the value negative one."
        self.play(self.lecture[3].animate.set_color("#FFD700"))
        
        # Fix for Issue 41: Label '-1' at grid D2
        label_neg_one = Text("-1", font_size=24, color="#FFD700")
        self.place_at_grid(label_neg_one, 'D2', scale_factor=0.7)
        
        self.play(Write(label_neg_one), dot.animate.scale(1.2).set_color("#FFD700"))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Line: "Thus, e to the i pi equals negative one."
        self.play(self.lecture[4].animate.set_color("#FFFFFF"))
        
        # Fix for Issue 39: Result formula at area A2-A5
        result_formula = Text("e^(i\u03c0) = -1", font_size=36, color=WHITE)
        self.place_in_area(result_formula, 'A2', 'A5', scale_factor=0.8)
        
        self.play(FadeIn(result_formula))
        self.wait(2)
