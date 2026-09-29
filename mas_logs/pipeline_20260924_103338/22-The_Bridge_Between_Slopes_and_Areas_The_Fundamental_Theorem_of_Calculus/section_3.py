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
        self.setup_layout("Fundamental Theorem of Calculus", [
            "Antiderivatives link slopes to total area.",
            "Subtract boundaries to find net change.",
            "It simplifies complex summation tasks."
        ])

        # Assets
        curve_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/curve.svg")
        graph_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/graph.svg")
        
        # Define axes
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 4, 1], axis_config={"include_tip": True})
        self.place_in_area(axes, "A1", "D4", scale_factor=0.55)
        
        func = lambda x: 0.25 * x**2
        curve = axes.plot(func, color="#8bc34a")
        
        # Integral label
        integral = MathTex(r"\int_a^b f(x) \, dx", color=WHITE)
        self.place_at_grid(integral, "E5", scale_factor=0.7)
        
        # Position icons
        self.place_at_grid(curve_icon, "A6", scale_factor=0.3)
        self.place_at_grid(graph_icon, "E2", scale_factor=0.3)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#8bc34a")
        self.play(FadeIn(curve_icon), Create(axes), Create(curve))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#fff176")
        
        # Area filler
        b_val = ValueTracker(3)
        area = always_redraw(lambda: axes.get_area(curve, x_range=[0, b_val.get_value()], color="#fff176", opacity=0.5))
        
        self.add(area)
        self.play(Write(integral), FadeIn(graph_icon))
        self.play(b_val.animate.set_value(4), run_time=2)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#ffffff")
        self.wait(1)
