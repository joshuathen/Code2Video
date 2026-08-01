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
        lecture_lines = [
            "Vectors aren't just arrows; they are members of sets.",
            "If it follows the rules, it is a vector.",
            "An arrow can morph into a quadratic function.",
            "Or even represent a digital RGB color code.",
            "This shift in perspective defines modern linear algebra."
        ]
        self.setup_layout("The Shift in Perspective", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Show a white arrow #FFFFFF and the text 'Set V' #00FF00 around it.
        self.lecture[0].set_color("#00FF00")
        
        # Define a container circle representing Set V
        set_v_boundary = Circle(radius=2.2, color="#00FF00", stroke_width=2)
        self.place_in_area(set_v_boundary, "A1", "F6")
        
        set_v_label = Text("Set V", color="#00FF00", font_size=24)
        self.place_at_grid(set_v_label, "A1")
        
        vector_arrow = Arrow(start=LEFT, end=RIGHT, color="#FFFFFF", buff=0)
        self.place_in_area(vector_arrow, "C3", "D4", scale_factor=1.2)
        
        self.play(
            Create(set_v_boundary),
            Write(set_v_label),
            GrowArrow(vector_arrow),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The arrow glows #FFFF00 while text 'Axioms' #FF00FF appears above it.
        self.lecture[1].set_color("#FF00FF")
        
        axioms_text = Text("Axioms", color="#FF00FF", font_size=32)
        self.place_at_grid(axioms_text, "B3")
        
        self.play(
            vector_arrow.animate.set_color("#FFFF00").set_stroke(width=10),
            Write(axioms_text),
            run_time=1
        )
        self.play(vector_arrow.animate.set_stroke(width=5), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The arrow #FFFFFF morphs into a quadratic curve #00FFFF with label 'f(x) = ax^2 + bx + c'.
        self.lecture[2].set_color("#00FFFF")
        
        # Create a quadratic curve
        curve = FunctionGraph(
            lambda x: 0.5 * x**2 - 0.5,
            x_range=[-1.5, 1.5],
            color="#00FFFF"
        )
        self.place_in_area(curve, "C3", "E4", scale_factor=0.8)
        
        # Fixed FileNotFoundError: [Errno 2] No such file or directory: 'latex'
        # Replaced MathTex with Text to avoid LaTeX dependency
        curve_label = Text("f(x) = ax² + bx + c", color="#00FFFF", font_size=24)
        self.place_at_grid(curve_label, "F3")
        
        self.play(
            FadeOut(axioms_text),
            ReplacementTransform(vector_arrow, curve),
            Write(curve_label),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Curve #00FFFF morphs into three colored boxes: Red #FF0000, Green #00FF00, Blue #0000FF.
        self.lecture[3].set_color("#FF0000")
        
        box_r = Square(side_length=0.8, fill_opacity=0.8, fill_color="#FF0000", stroke_color=WHITE)
        box_g = Square(side_length=0.8, fill_opacity=0.8, fill_color="#00FF00", stroke_color=WHITE)
        box_b = Square(side_length=0.8, fill_opacity=0.8, fill_color="#0000FF", stroke_color=WHITE)
        
        val_r = Text("255", font_size=18).move_to(box_r.get_center())
        val_g = Text("128", font_size=18).move_to(box_g.get_center())
        val_b = Text("0", font_size=18).move_to(box_b.get_center())
        
        rgb_group = VGroup(
            VGroup(box_r, val_r),
            VGroup(box_g, val_g),
            VGroup(box_b, val_b)
        ).arrange(RIGHT, buff=0.2)
        
        self.place_in_area(rgb_group, "C2", "D5")
        
        self.play(
            FadeOut(curve_label),
            ReplacementTransform(curve, rgb_group),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # The boxes fade out, leaving the text 'Abstract Vector Space' #FFFFFF centered on screen.
        self.lecture[4].set_color("#FFFFFF")
        
        abstract_text = Text("Abstract Vector Space", color="#FFFFFF", font_size=40)
        self.place_in_area(abstract_text, "A1", "F6")
        
        self.play(
            FadeOut(rgb_group),
            FadeOut(set_v_boundary),
            FadeOut(set_v_label),
            Write(abstract_text),
            run_time=1.5
        )
        self.wait(2)
