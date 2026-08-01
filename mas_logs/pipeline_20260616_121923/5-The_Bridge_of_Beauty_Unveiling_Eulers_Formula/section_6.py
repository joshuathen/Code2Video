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

class Section6Scene(TeachingScene):
    def construct(self):
        # Initialize title and lecture lines
        lecture_lines = [
            "Thus, e to the i-pi equals negative one.",
            "Adding one brings the entire system to zero.",
            "e to the i-pi plus one equals zero.",
            "Growth, rotation, and cycles merge in harmony.",
            "Five strangers are now one unified truth."
        ]
        self.setup_layout("The Final Equation", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Equation construction: e^{iπ} = -1
        # Using Text objects for reliability and consistent layout
        e1 = Text("e", font_size=40)
        i1 = Text("i", font_size=24).move_to(e1.get_top() + RIGHT*0.2 + UP*0.1)
        pi1 = Text("π", font_size=24).next_to(i1, RIGHT, buff=0.05)
        eq1 = Text("=", font_size=40).next_to(e1, RIGHT, buff=0.9)
        minus1 = Text("-", font_size=40).next_to(eq1, RIGHT, buff=0.2)
        one1 = Text("1", font_size=40).next_to(minus1, RIGHT, buff=0.1)
        formula_1 = VGroup(e1, i1, pi1, eq1, minus1, one1)
        
        # Resolving Issue 46: Positioning the equation in a clear area
        self.place_in_area(formula_1, 'B2', 'B5', scale_factor=1.2)
        
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(Write(formula_1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(BLUE))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN))
        
        # New equation construction: e^{iπ} + 1 = 0
        e2 = Text("e", font_size=40)
        i2 = Text("i", font_size=24).move_to(e2.get_top() + RIGHT*0.2 + UP*0.1)
        pi2 = Text("π", font_size=24).next_to(i2, RIGHT, buff=0.05)
        plus2 = Text("+", font_size=40).next_to(e2, RIGHT, buff=0.7)
        one2 = Text("1", font_size=40).next_to(plus2, RIGHT, buff=0.2)
        eq2 = Text("=", font_size=40).next_to(one2, RIGHT, buff=0.2)
        zero2 = Text("0", font_size=40).next_to(eq2, RIGHT, buff=0.2)
        formula_2 = VGroup(e2, i2, pi2, plus2, one2, eq2, zero2)
        
        self.place_in_area(formula_2, 'B2', 'B5', scale_factor=1.2)
        
        self.play(ReplacementTransform(formula_1, formula_2))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(PURPLE))
        
        # Complex plane visualization
        # Resolving Issue 47: Positioning the plane to avoid clutter
        plane = ComplexPlane(
            x_range=[-2, 2, 1], y_range=[-2, 2, 1],
            x_length=3.5, y_length=3.5,
            background_line_style={"stroke_opacity": 0.3}
        )
        self.place_at_grid(plane, 'E4', scale_factor=1.2)
        
        radius_val = plane.coords_to_point(1, 0)[0] - plane.coords_to_point(0, 0)[0]
        circle = Circle(radius=radius_val, color=WHITE, stroke_opacity=0.4)
        circle.move_to(plane.get_origin())
        
        dot = Dot(plane.coords_to_point(1, 0), color=YELLOW)
        arc = Arc(radius=radius_val, start_angle=0, angle=PI, color=YELLOW, arc_center=plane.get_origin())
        
        self.play(Create(plane), Create(circle))
        self.play(
            dot.animate.move_to(plane.coords_to_point(-1, 0)),
            Create(arc),
            run_time=2
        )
        
        # Highlight colors for parts of the final identity
        self.play(
            formula_2[0].animate.set_color(YELLOW),     # e (growth)
            formula_2[1].animate.set_color(LIGHT_PINK), # i (rotation)
            formula_2[2].animate.set_color(ORANGE),     # pi (circles)
            formula_2[4].animate.set_color(BLUE),       # 1 (unit)
            formula_2[6].animate.set_color(GOLD),       # 0 (nothingness)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(GOLD))
        
        # Create a dimming effect and highlight the final beauty of the formula
        dim_rect = Rectangle(width=14, height=8, fill_color=BLACK, fill_opacity=0.75, stroke_width=0)
        final_title = Text("The Bridge of Beauty", font_size=32, color=WHITE).to_edge(UP)
        
        self.play(
            FadeIn(dim_rect),
            ReplacementTransform(self.title, final_title),
            formula_2.animate.scale(1.2).set_stroke(width=2, opacity=1),
            run_time=2
        )
        self.wait(3)
