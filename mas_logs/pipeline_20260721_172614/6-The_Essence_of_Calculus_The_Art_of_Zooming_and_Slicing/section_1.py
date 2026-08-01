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

class Section1Scene(TeachingScene):
    def construct(self):
        # Fetching title and lecture lines from storyboard
        title_text = "The Static vs. The Dynamic"
        lecture_lines = [
            "Algebra handles constant speeds and straight lines perfectly.",
            "But the real world is curvy and ever-changing.",
            "Calculus is the lens for studying change as it happens."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Show a blue line (#0000FF) and a car moving steadily along it.
        # Label the line 'Algebra: Constant Speed' in #ADD8E6.
        
        blue_line = Line(self.grid["C1"], self.grid["C6"], color="#0000FF")
        car = Triangle(color="#ADD8E6", fill_opacity=1).rotate(-PI/2)
        # Issue 27 Fix: Increase scale factor to 0.4
        self.place_at_grid(car, "C1", scale_factor=0.4)
        
        algebra_label = Text("Algebra: Constant Speed", font_size=24, color="#ADD8E6")
        # Issue 25 Fix: Place in area B2 to B5 for better centering
        self.place_in_area(algebra_label, "B2", "B5", scale_factor=0.7)
        
        self.play(self.lecture[0].animate.set_color("#ADD8E6"))
        self.play(Create(blue_line), Write(algebra_label))
        self.play(car.animate.move_to(self.grid["C6"]), run_time=3, rate_func=linear)
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        # Fade out the car, show a curved yellow path (#FFFF00).
        # A cheetah sprite (#FFFFE0) moves along the curve with varying speed.
        
        curvy_path = VMobject(color="#FFFF00")
        # Define a path using grid points
        path_points = [
            self.grid["D1"],
            self.grid["E2"],
            self.grid["D3"],
            self.grid["F4"],
            self.grid["E5"],
            self.grid["D6"]
        ]
        curvy_path.set_points_as_corners(path_points).make_smooth()
        
        cheetah = Dot(color="#FFFFE0", radius=0.15)
        cheetah_label = Text("Real World: Dynamic Change", font_size=24, color="#FFFFE0")
        # Issue 26 Fix: Place in area F2 to F5 for better centering and boundary safety
        self.place_in_area(cheetah_label, "F2", "F5", scale_factor=0.7)

        self.play(
            FadeOut(car),
            FadeOut(blue_line),
            FadeOut(algebra_label),
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FFFFE0")
        )
        self.play(Create(curvy_path), Write(cheetah_label))
        
        # Varying speed: slow -> fast -> slow
        self.play(MoveAlongPath(cheetah, curvy_path), run_time=4, rate_func=slow_into)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # A magnifying glass icon (#FFFFFF) zooms in on the curve.
        # Calculus is the lens for studying change as it happens.
        
        # Issue 40 Fix: Use the SVG asset for the magnifying glass
        magnifying_glass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magn.svg")
        magnifying_glass.set_color(WHITE)
        
        # Place it near the center of the curve
        self.place_at_grid(magnifying_glass, "D3", scale_factor=0.8)
        
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FFFFFF")
        )
        self.play(FadeIn(magnifying_glass))
        
        # Zoom effect animation
        self.play(
            magnifying_glass.animate.scale(1.5).move_to(self.grid["D3"] + RIGHT*0.2),
            cheetah.animate.set_opacity(0.3),
            run_time=1.5
        )
        
        # Show a "zoomed-in" view: a straight segment representing a local linear approximation
        tangent_segment = Line(
            self.grid["D3"] + LEFT*0.5 + DOWN*0.2,
            self.grid["D3"] + RIGHT*0.5 + UP*0.2,
            color=WHITE, stroke_width=6
        )
        self.play(Create(tangent_segment))
        self.wait(2)
