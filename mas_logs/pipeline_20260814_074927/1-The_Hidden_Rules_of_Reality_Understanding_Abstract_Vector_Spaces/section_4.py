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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Functions can act exactly like geometric vectors.",
            "Adding two functions creates a new, combined function.",
            "Scaling a function stretches its graph vertically.",
            "The zero function acts as the origin point.",
            "This makes continuous functions members of a vector space."
        ]
        self.setup_layout("Abstract Example 1: The Space of Functions", lecture_lines)
        
        # Set initial transparency for lecture lines
        for line in self.lecture:
            line.set_opacity(0.3)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_opacity(1.0).set_color(WHITE))
        
        # Divider line between col 3 and 4
        divider_top = (self.grid["A3"] + self.grid["A4"]) / 2 + UP * 0.5
        divider_bottom = (self.grid["F3"] + self.grid["F4"]) / 2 + DOWN * 0.5
        divider = Line(divider_top, divider_bottom, color=WHITE)
        
        # Vector Origin in the left area (cols 1-3)
        v_origin = self.grid["D2"] + LEFT * 0.3 + DOWN * 0.3
        
        u = Vector([0.8, 0.4, 0], color=BLUE)
        u.shift(v_origin - u.get_start())
        
        v = Vector([0.4, 0.8, 0], color=YELLOW)
        v.shift(u.get_end() - v.get_start())
        
        u_plus_v = Vector(u.get_vector() + v.get_vector(), color=WHITE)
        u_plus_v.shift(v_origin - u_plus_v.get_start())
        
        label_u = MathTex("\\vec{u}", color=BLUE, font_size=20).next_to(u, LEFT, buff=0.1)
        label_v = MathTex("\\vec{v}", color=YELLOW, font_size=20).next_to(v, RIGHT, buff=0.1)
        
        # Right Side: Empty Axes (cols 4-6)
        axes = Axes(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            x_length=3.5,
            y_length=3.5,
            axis_config={"include_tip": True, "color": WHITE}
        )
        self.place_in_area(axes, "B4", "E6")
        
        self.play(Create(divider))
        self.play(Create(u), Write(label_u), run_time=1)
        self.play(Create(v), Write(label_v), run_time=1)
        self.play(Create(u_plus_v), run_time=1)
        self.play(Create(axes))
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_opacity(0.3),
            self.lecture[1].animate.set_opacity(1.0).set_color(WHITE)
        )
        
        f_func = lambda x: np.sin(PI * x)
        g_func = lambda x: 0.5 * np.cos(2 * PI * x)
        
        f_graph = axes.plot(f_func, color="#00FFFF")
        g_graph = axes.plot(g_func, color="#FFD700")
        h_graph = axes.plot(lambda x: f_func(x) + g_func(x), color=WHITE)
        
        f_label = MathTex("f(x)", color="#00FFFF", font_size=20).next_to(f_graph, UP, buff=0.1)
        g_label = MathTex("g(x)", color="#FFD700", font_size=20).next_to(g_graph, DOWN, buff=0.1)
        h_label = MathTex("(f+g)(x)", color=WHITE, font_size=20).next_to(h_graph, RIGHT, buff=0.1)

        self.play(Create(f_graph), Write(f_label))
        self.play(Create(g_graph), Write(g_label))
        self.wait(0.5)
        self.play(Create(h_graph), Write(h_label))
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_opacity(0.3),
            self.lecture[2].animate.set_opacity(1.0).set_color(WHITE)
        )
        
        # Scaling demonstration
        self.play(h_graph.animate.stretch(1.5, dim=1, about_point=axes.c2p(0,0,0)), run_time=1.5)
        self.play(h_graph.animate.stretch(0.5/1.5, dim=1, about_point=axes.c2p(0,0,0)), run_time=1.5)
        self.play(h_graph.animate.stretch(1.0/0.5, dim=1, about_point=axes.c2p(0,0,0)), run_time=1)
        self.wait(1)
        
        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_opacity(0.3),
            self.lecture[3].animate.set_opacity(1.0).set_color("#888888")
        )
        
        zero_func_graph = axes.plot(lambda x: 0, color="#888888", stroke_width=4)
        zero_label = MathTex("0(x) = 0", color="#888888", font_size=20).next_to(zero_func_graph, DOWN, buff=0.1)
        
        self.play(Create(zero_func_graph), Write(zero_label))
        self.wait(2)
        
        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_opacity(0.3),
            self.lecture[4].animate.set_opacity(1.0).set_color(WHITE)
        )
        
        final_concept = Text("Vector Space Rules Apply", font_size=20, color=WHITE)
        self.place_at_grid(final_concept, "F5")
        
        self.play(Write(final_concept))
        self.play(Indicate(final_concept))
        self.wait(3)
