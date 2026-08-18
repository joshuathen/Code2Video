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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Transforming the Problem into Geometry", [
            "Let's simplify the physics using a clever mathematical trick.",
            "We rescale velocities by the square root of mass.",
            "This transforms the complex energy ellipse into a circle.",
            "Now, every collision is a point on this circle.",
            "The physics problem has become a geometry problem."
        ])

        # Colors
        ELLIPSE_COLOR = "#58C4DD" # Sky Blue
        CIRCLE_COLOR = "#ADFF2F"  # GreenYellow
        FORMULA_COLOR = "#F0E68C" # Khaki

        # === Animation for Lecture Line 1 ===
        # "Let's simplify the physics using a clever mathematical trick."
        self.play(self.lecture[0].animate.set_color(FORMULA_COLOR))
        
        energy_formula = MathTex(
            r"E = \frac{1}{2} m v^2 + \frac{1}{2} M V^2",
            color=FORMULA_COLOR
        )
        # Issue 27/40: Position in A2-A5, scale 0.8
        self.place_in_area(energy_formula, "A2", "A5", scale_factor=0.8)
        self.play(Write(energy_formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "We rescale velocities by the square root of mass."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(WHITE)
        )
        
        substitution = MathTex(
            r"V^* = V \sqrt{M}",
            color=WHITE
        )
        # Issue 28/40: Position in B2-B5, scale 0.8
        self.place_in_area(substitution, "B2", "B5", scale_factor=0.8)
        
        # Transitioning the formula
        new_energy_formula = MathTex(
            r"E = \frac{1}{2} m v^2 + \frac{1}{2} (V^*)^2",
            color=FORMULA_COLOR
        )
        self.place_in_area(new_energy_formula, "A2", "A5", scale_factor=0.8)

        self.play(
            FadeIn(substitution, shift=DOWN*0.2),
            Transform(energy_formula, new_energy_formula)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "This transforms the complex energy ellipse into a circle."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(CIRCLE_COLOR)
        )

        # Setup axes
        # Issue 29/40: Expand to C1-F6, scale 0.9
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=5,
            y_length=4,
            axis_config={"include_tip": True, "color": WHITE}
        )
        self.place_in_area(axes, "C1", "F6", scale_factor=0.9)
        
        v_label = MathTex("v", font_size=24).next_to(axes.x_axis.get_end(), RIGHT, buff=0.1)
        V_label = MathTex("V", font_size=24).next_to(axes.y_axis.get_top(), UP, buff=0.1)
        
        # Ellipse (representing conservation of energy in v, V space)
        # Stretched horizontally because M >> m
        ellipse = Ellipse(width=4.0, height=1.5, color=ELLIPSE_COLOR)
        ellipse.move_to(axes.c2p(0, 0))

        self.play(Create(axes), Write(v_label), Write(V_label))
        self.play(Create(ellipse))
        self.wait(1)

        # Transformation to circle
        circle = Circle(radius=1.5, color=CIRCLE_COLOR)
        circle.move_to(axes.c2p(0, 0))
        
        # New axis label
        V_star_label = MathTex("V^*", font_size=24, color=CIRCLE_COLOR)
        V_star_label.next_to(axes.y_axis.get_top(), UP, buff=0.1)

        self.play(
            ReplacementTransform(ellipse, circle),
            Transform(V_label, V_star_label),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "Now, every collision is a point on this circle."
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(WHITE)
        )

        # Add a few points on the circle
        points = VGroup(*[
            Dot(circle.point_at_angle(angle), color=YELLOW, radius=0.06)
            for angle in [PI/3, 2*PI/3, 4*PI/3, 5*PI/3]
        ])
        
        self.play(Create(points))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "The physics problem has become a geometry problem."
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(WHITE)
        )

        # Pulse the circle to emphasize the geometry
        self.play(
            circle.animate.scale(1.1),
            rate_func=there_and_back,
            run_time=0.5
        )
        self.play(
            circle.animate.scale(1.1),
            rate_func=there_and_back,
            run_time=0.5
        )
        
        self.wait(2)
