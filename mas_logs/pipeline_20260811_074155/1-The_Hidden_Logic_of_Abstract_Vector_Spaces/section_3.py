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
        lecture_lines = [
            "- Vectors don't have to look like physical arrows.",
            "- Polynomials can function exactly like traditional vectors.",
            "- Adding curves is done point-wise to create new sums.",
            "- We can build any polynomial using simple basis functions.",
            "- These basis functions span the entire polynomial space."
        ]
        self.setup_layout("Strange Vectors: Polynomials and Functions", lecture_lines)

        # Colors
        COLOR_F = "#FF4500" # Orange
        COLOR_G = "#ADFF2F" # Lime
        COLOR_H = "#00BFFF" # Blue
        COLOR_BASIS = "#DA70D6" # Purple

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # Plot orange parabola f(x)=x^2 and lime line g(x)=2x
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_F)
        )
        
        axes = Axes(
            x_range=[-1, 3, 1],
            y_range=[-1, 8, 2],
            x_length=4.5,
            y_length=4.5,
            axis_config={"color": GREY, "include_numbers": False},
            tips=False
        )
        # Fix 23: scale axes to 0.8 and move to B1-F6
        self.place_in_area(axes, "B1", "F6", scale_factor=0.8)
        
        f_graph = axes.plot(lambda x: x**2, x_range=[-1, 2.4], color=COLOR_F)
        f_label = MathTex("f(x)=x^2", color=COLOR_F, font_size=24)
        self.place_at_grid(f_label, "B6", scale_factor=0.8)
        
        g_graph = axes.plot(lambda x: 2*x, x_range=[-0.5, 2.8], color=COLOR_G)
        g_label = MathTex("g(x)=2x", color=COLOR_G, font_size=24)
        # Fix 25: move g_label to E5 and scale 0.8
        self.place_at_grid(g_label, "E5", scale_factor=0.8)

        self.play(Create(axes), Create(f_graph), Create(g_graph), Write(f_label), Write(g_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Point-wise addition
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_H)
        )
        
        x_val = 1.2
        point_f = axes.c2p(x_val, x_val**2)
        point_g = axes.c2p(x_val, 2*x_val)
        point_h = axes.c2p(x_val, x_val**2 + 2*x_val)
        
        dot_f = Dot(point_f, color=COLOR_F, radius=0.06)
        dot_g = Dot(point_g, color=COLOR_G, radius=0.06)
        dot_h = Dot(point_h, color=COLOR_H, radius=0.06)
        
        seg_f = Line(axes.c2p(x_val, 0), point_f, color=COLOR_F, stroke_width=4)
        seg_g = Line(axes.c2p(x_val, 0), point_g, color=COLOR_G, stroke_width=4)
        
        self.play(FadeIn(dot_f), FadeIn(dot_g))
        self.play(Create(seg_f), Create(seg_g))
        self.wait(0.5)
        
        # Move seg_g to be stacked on seg_f
        target_seg_g = Line(point_f, point_h, color=COLOR_G, stroke_width=4)
        self.play(
            seg_g.animate.move_to(target_seg_g.get_center()),
            FadeIn(dot_h)
        )
        self.wait(1)
        
        h_graph = axes.plot(lambda x: x**2 + 2*x, x_range=[-0.5, 2.0], color=COLOR_H)
        h_label = MathTex("h(x)=x^2+2x", color=COLOR_H, font_size=24)
        self.place_at_grid(h_label, "C6", scale_factor=0.8)
        
        self.play(
            ReplacementTransform(f_graph.copy(), h_graph),
            ReplacementTransform(g_graph.copy(), h_graph),
            FadeOut(f_label, g_label, seg_f, seg_g, dot_f, dot_g, dot_h),
            Write(h_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Basis set {1, x, x^2} in a purple box
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(COLOR_BASIS)
        )
        
        basis_set = MathTex(r"\{1, x, x^2\}", color=COLOR_BASIS, font_size=32)
        box = SurroundingRectangle(basis_set, color=COLOR_BASIS, buff=0.2)
        basis_group = VGroup(basis_set, box)
        # Fix 24: basis_group at A4 with scale 0.8
        self.place_at_grid(basis_group, "A4", scale_factor=0.8)
        
        self.play(Write(basis_group))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Flash basis functions
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(COLOR_BASIS)
        )
        
        b1 = axes.plot(lambda x: 1, x_range=[-1, 3], color=WHITE)
        b2 = axes.plot(lambda x: x, x_range=[-1, 3], color=WHITE)
        b3 = axes.plot(lambda x: x**2, x_range=[-1, 3], color=WHITE)
        
        for b in [b1, b2, b3]:
            self.play(Create(b), run_time=0.6)
            self.play(FadeOut(b), run_time=0.4)
        
        self.play(Indicate(h_graph, color=COLOR_BASIS))
        self.wait(2)
