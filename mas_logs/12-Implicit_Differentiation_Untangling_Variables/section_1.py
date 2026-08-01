from manim import *
import numpy as np

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        self.lecture = lecture_lines
        # Setup logic would typically initialize self.lecture_vgroup as a VGroup of Tex objects
        pass

    def construct(self):
        # ... (previous code)
        
        # === Animation for Lecture Line 1 ===
        # Color line 1 to match the 'f(x)' box
        # FIX: Changed self.lecture[0] (a string) to self.lecture_vgroup[0] ( the Mobject)
        self.play(self.lecture_vgroup[0].animate.set_color("#ADD8E6"))
        
        explicit_eq = MathTex("y = x^2", color="#FFFFFF")
        self.place_at_grid(explicit_eq, "A3", scale_factor=1.0)
        self.lecture = lecture_lines
        # Implementation of setup_layout
        pass

    def construct(self):
        # ... existing logic ...
        # Changed MathTex to Text to avoid FileNotFoundError: 'latex' when LaTeX is not installed
        explicit_eq = Text("y = x^2", color="#FFFFFF")
        # ... existing logic ...
        self.title = Text(title_text).to_edge(UP)
        # Convert raw strings to Text mobjects so they have the .animate attribute
        self.lecture = VGroup(*[Text(line) for line in lecture_lines]).arrange(DOWN).next_to(self.title, DOWN)
        self.add(self.title, self.lecture)

    def place_at_grid(self, mobject, cell, scale_factor=1.0):
        mobject.scale(scale_factor)
        self.add(mobject)

    def construct(self):
        # Sample data to match the traceback context
        lecture_data = ["Implicit Differentiation", "Untangling Variables"]
        self.setup_layout("Lecture Title", lecture_data)

        # === Animation for Lecture Line 1 ===
        # Color line 1 to match the 'f(x)' box
        # This works now because self.lecture[0] is a Text object, not a string
        self.play(self.lecture[0].animate.set_color("#ADD8E6"))

        explicit_eq = MathTex("y = x^2", color="#FFFFFF")
        self.place_at_grid(explicit_eq, "A3", scale_factor=1.0)
        self.lecture = lecture_lines
        # Store title and create a VGroup of Text mobjects for the lecture lines
        self.title = Text(title_text)
        self.lecture_lines = VGroup(*[Text(line) for line in lecture_lines])
        
    def construct(self):
        # ... existing code ...
        
        # === Animation for Lecture Line 1 ===
        # Color line 1 to match the 'f(x)' box
        # FIX: Use self.lecture_lines (the VGroup of Mobjects) instead of self.lecture (the list of strings)
        self.play(self.lecture_lines[0].animate.set_color("#ADD8E6"))
        
        explicit_eq = MathTex("y = x^2", color="#FFFFFF")
        self.place_at_grid(explicit_eq, "A3", scale_factor=1.0)
        self.title = Text(title_text).to_edge(UP)
        self.lecture = VGroup(*[Text(line) for line in lecture_lines]).next_to(self.title, DOWN)
        self.add(self.title, self.lecture)
        self.lecture = lecture_lines
        pass

    def construct(self):
        # Color line 1 to match the 'f(x)' box
        self.play(self.lecture[0].animate.set_color("#ADD8E6"))
        
        # Changed MathTex to Text to avoid the FileNotFoundError for 'latex'
        explicit_eq = Text("y = x^2", color="#FFFFFF")
        self.place_at_grid(explicit_eq, "A3", scale_factor=1.0)
        
        f_box = RoundedRectangle(corner_radius=0.1, height=0.8, width=1.2, color="#ADD8E6")
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
        # Setup layout
        title_text = "Introduction: Explicit vs. Implicit"
        lecture_lines = [
            "In explicit functions, y is isolated and easy to find.",
            "In implicit relations, x and y are tangled together.",
            "Implicit differentiation is needed when solving for y is hard."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Color line 1 to match the 'f(x)' box
        self.play(self.lecture[0].animate.set_color("#ADD8E6"))
        
        explicit_eq = MathTex("y = x^2", color="#FFFFFF")
        self.place_at_grid(explicit_eq, "A3", scale_factor=1.0)
        
        f_box = RoundedRectangle(corner_radius=0.1, height=0.8, width=1.2, color="#ADD8E6")
        self.place_at_grid(f_box, "B3")
        f_label = Text("f(x)", color="#ADD8E6", font_size=20)
        self.place_at_grid(f_label, "B3")
        
        x_label = Text("x", color=WHITE, font_size=24)
        self.place_at_grid(x_label, "B1")
        
        y_label = Text("y", color=WHITE, font_size=24)
        self.place_at_grid(y_label, "B5")
        
        # Arrows using grid positions
        arrow_in = Arrow(start=self.grid["B1"] + RIGHT*0.3, end=self.grid["B3"] + LEFT*0.7, color=WHITE, buff=0)
        arrow_out = Arrow(start=self.grid["B3"] + RIGHT*0.7, end=self.grid["B5"] + LEFT*0.3, color=WHITE, buff=0)
        
        self.play(
            Write(explicit_eq),
            Create(f_box),
            Write(f_label)
        )
        self.play(
            Write(x_label),
            GrowArrow(arrow_in)
        )
        self.play(
            Write(y_label),
            GrowArrow(arrow_out)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Color line 2 to match the knot
        self.play(self.lecture[1].animate.set_color("#D3D3D3"))
        
        implicit_eq = MathTex("x^2 + y^2 = 25", color="#FFFFFF")
        self.place_at_grid(implicit_eq, "C3", scale_factor=1.0)
        
        # Knot graphic - a messy squiggle
        knot_path = ParametricFunction(
            lambda t: np.array([
                0.6 * np.cos(t) + 0.2 * np.cos(3*t),
                0.6 * np.sin(t) + 0.2 * np.sin(2*t),
                0
            ]), t_range=[0, 4*PI], color="#D3D3D3"
        )
        self.place_at_grid(knot_path, "E3", scale_factor=1.2)
        
        x_circ = Circle(radius=0.25, color="#FFB6C1", fill_opacity=0.3)
        x_text = Text("x", color="#FFB6C1", font_size=18)
        x_group = VGroup(x_circ, x_text)
        self.place_at_grid(x_group, "E2")
        
        y_circ = Circle(radius=0.25, color="#90EE90", fill_opacity=0.3)
        y_text = Text("y", color="#90EE90", font_size=18)
        y_group = VGroup(y_circ, y_text)
        self.place_at_grid(y_group, "E4")
        
        self.play(Write(implicit_eq))
        self.play(Create(knot_path))
        self.play(FadeIn(x_group), FadeIn(y_group))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Color line 3 to match the question mark
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        
        q_mark = Text("?", color="#FFFF00", font_size=48)
        self.place_at_grid(q_mark, "E3")
        
        # Gear construction
        gear_core = Circle(radius=0.4, color=WHITE)
        teeth = VGroup(*[
            Rectangle(width=0.15, height=0.2, color=WHITE, fill_opacity=1)
            .move_to(np.array([0.45 * np.cos(a), 0.45 * np.sin(a), 0]))
            .rotate(a)
            for a in np.linspace(0, TAU, 8, endpoint=False)
        ])
        gear = VGroup(gear_core, teeth)
        self.place_at_grid(gear, "E3", scale_factor=0.8)
        
        self.play(Write(q_mark))
        self.wait(0.5)
        
        # Transform the knot and labels into the gear
        self.play(
            FadeOut(q_mark),
            FadeOut(x_group),
            FadeOut(y_group),
            ReplacementTransform(knot_path, gear)
        )
        self.play(Rotate(gear, angle=2*PI), run_time=2)
        self.wait(2)
