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
        # Initial layout setup
        self.setup_layout("The Shortcut: L'H\u00f4pital's Rule", [
            "L'Hopital's rule compares the functions' individual growth rates.",
            "The ratio of functions equals the ratio of derivatives.",
            "Zooming in, curves look like straight tangent lines.",
            "The ratio of their slopes reveals the limit value.",
            "This shortcut turns an indeterminate mess into clarity."
        ])

        # Define consistent colors
        COLOR_SIN = "#00AAFF"   # Blue
        COLOR_X = "#FFAA00"     # Orange
        COLOR_GREEN = "#00FF00" # Green
        COLOR_WHITE = "#FFFFFF" # White

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_SIN))
        
        axes = Axes(
            x_range=[-1.5, 1.5, 0.5],
            y_range=[-1.5, 1.5, 0.5],
            axis_config={"include_tip": False, "color": GREY_C},
            tips=False
        )
        sin_curve = axes.plot(lambda x: np.sin(x), color=COLOR_SIN, x_range=[-1.5, 1.5])
        x_line = axes.plot(lambda x: x, color=COLOR_X, x_range=[-1.5, 1.5])
        
        graph_group = VGroup(axes, sin_curve, x_line)
        self.place_in_area(graph_group, "C2", "E5", scale_factor=0.6)
        
        self.play(Create(axes), Create(sin_curve), Create(x_line))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Replaced MathTex with Text to avoid FileNotFoundError: 'latex'
        self.play(self.lecture[1].animate.set_color(COLOR_WHITE))
        
        formula = Text(
            "lim[x→0] f(x)/g(x) = lim[x→0] f'(x)/g'(x)",
            font_size=20,
            color=COLOR_WHITE
        )
        self.place_in_area(formula, "A2", "A5", scale_factor=0.8)
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(COLOR_WHITE))
        
        origin_point = axes.c2p(0, 0)
        self.play(
            graph_group.animate.scale(4, about_point=origin_point),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(COLOR_GREEN))
        
        # Triangle 1 for positive x
        t1_p1 = axes.c2p(0, 0)
        t1_p2 = axes.c2p(0.3, 0)
        t1_p3 = axes.c2p(0.3, 0.3)
        tri1 = Polygon(t1_p1, t1_p2, t1_p3, color=COLOR_GREEN, stroke_width=2, fill_opacity=0.3)
        
        # Triangle 2 for negative x
        t2_p1 = axes.c2p(0, 0)
        t2_p2 = axes.c2p(-0.3, 0)
        t2_p3 = axes.c2p(-0.3, -0.3)
        tri2 = Polygon(t2_p1, t2_p2, t2_p3, color=COLOR_GREEN, stroke_width=2, fill_opacity=0.3)
        
        label_slope1 = Text("slope = 1", font_size=18, color=COLOR_GREEN)
        self.place_at_grid(label_slope1, "B4", scale_factor=1.0)
        
        label_slope2 = Text("slope = 1", font_size=18, color=COLOR_GREEN)
        self.place_at_grid(label_slope2, "F4", scale_factor=1.0)
        
        self.play(Create(tri1), Create(tri2))
        self.play(Write(label_slope1), Write(label_slope2))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Replaced
