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
        self.setup_layout("The Algebra-Geometry Bridge", [
            "Linear systems can be seen as vector combinations.",
            "We seek scalars x and y for vector W.",
            "W equals x times A plus y times B.",
            "This transforms algebra into a geometric puzzle.",
            "We must find how to reach the target destination."
        ])

        # Colors
        color_a = "#00FF00"  # Green
        color_b = "#0000FF"  # Blue
        color_w = "#FF0000"  # Red
        color_scalar = "#FFFF00"  # Yellow
        color_equation = "#FFFFFF"  # White

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(color_equation))
        
        system = MathTex(
            r"\begin{cases} a_1 x + b_1 y = w_1 \\ a_2 x + b_2 y = w_2 \end{cases}",
            color=color_equation
        )
        self.place_in_area(system, "A2", "B6", scale_factor=1.2)
        self.play(Write(system))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(color_scalar))
        
        # Highlight x and y in the system using set_color_by_tex for safety
        system_highlighted = MathTex(
            r"\begin{cases} a_1 x + b_1 y = w_1 \\ a_2 x + b_2 y = w_2 \end{cases}",
            color=color_equation
        )
        system_highlighted.set_color_by_tex("x", color_scalar)
        system_highlighted.set_color_by_tex("y", color_scalar)
        
        self.place_in_area(system_highlighted, "A2", "B6", scale_factor=1.2)
        self.play(Transform(system, system_highlighted))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(WHITE))
        
        vector_eq = MathTex(
            "x", r"\vec{A}", "+", "y", r"\vec{B}", "=", r"\vec{W}",
            tex_to_color_map={
                "x": color_scalar,
                "y": color_scalar,
                r"\vec{A}": color_a,
                r"\vec{B}": color_b,
                r"\vec{W}": color_w
            }
        )
        self.place_in_area(vector_eq, "A2", "B6", scale_factor=1.2)
        self.play(Transform(system, vector_eq))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#00FFFF"))
        
        # Grid and Vectors
        plane_anchor = Dot(radius=0)
        self.place_in_area(plane_anchor, "C2", "F6")
        
        plane = NumberPlane(
            x_range=[-1, 5, 1],
            y_range=[-1, 4, 1],
            x_length=4.5,
            y_length=3.5,
            background_line_style={"stroke_opacity": 0.4}
        ).move_to(plane_anchor.get_center())
        
        self.play(Create(plane))

        # Vector coordinates
        vec_a_coords = np.array([1, 1.5, 0])
        vec_b_coords = np.array([1.5, 0.5, 0])
        x_val, y_val = 1, 2 
        vec_w_coords = x_val * vec_a_coords + y_val * vec_b_coords

        vec_a = Arrow(plane.c2p(0, 0, 0), plane.c2p(*vec_a_coords), buff=0, color=color_a)
        vec_b = Arrow(plane.c2p(0, 0, 0), plane.c2p(*vec_b_coords), buff=0, color=color_b)
        vec_w = Arrow(plane.c2p(0, 0, 0), plane.c2p(*vec_w_coords), buff=0, color=color_w)
        
        label_a = MathTex(r"\vec{A}", color=color_a, font_size=24).next_to(vec_a.get_end(), UP, buff=0.1)
        label_b = MathTex(r"\vec{B}", color=color_b, font_size=24).next_to(vec_b.get_end(), RIGHT, buff=0.1)
        label_w = MathTex(r"\vec{W}", color=color_w, font_size=24).next_to(vec_w.get_end(), UR, buff=0.1)

        self.play(GrowArrow(vec_a), Write(label_a))
        self.play(GrowArrow(vec_b), Write(label_b))
        self.play(GrowArrow(vec_w), Write(label_w))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(color_w))
        
        # Tip-to-tail sum
        scaled_a = Arrow(plane.c2p(0, 0, 0), plane.c2p(*(x_val * vec_a_coords)), buff=0, color=color_a)
        scaled_b = Arrow(plane.c2p(*(x_val * vec_a_coords)), plane.c2p(*vec_w_coords), buff=0, color=color_b)
        
        label_xa = MathTex(r"x\vec{A}", color=color_a, font_size=24).next_to(scaled_a.get_center(), UL, buff=0.1)
        label_yb = MathTex(r"y\vec{B}", color=color_b, font_size=24).next_to(scaled_b.get_center(), DR, buff=0.1)

        self.play(
            FadeOut(vec_a), FadeOut(label_a),
            FadeOut(vec_b), FadeOut(label_b),
            ReplacementTransform(vec_a.copy(), scaled_a),
            Write(label_xa)
        )
        self.play(
            ReplacementTransform(vec_b.copy(), scaled_b),
            Write(label_yb)
        )
        self.wait(2)
