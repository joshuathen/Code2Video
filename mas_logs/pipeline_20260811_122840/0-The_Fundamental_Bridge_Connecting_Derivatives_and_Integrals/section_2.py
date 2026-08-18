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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Review of Prerequisite: The Derivative as a 'Zoom'", [
            "Derivatives measure the instantaneous rate of change.",
            "Imagine zooming into a curve to see slope.",
            "Slope defines the rate at that specific moment."
        ])
        
        # Reveal lecture lines
        self.lecture.set_opacity(1)
        
        # Load asset
        magnifier = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifyingglass.svg")
        
        # === Animation for Lecture Line 1 ===
        # Derivatives measure the instantaneous rate of change.
        self.lecture[0].set_color(BLUE)
        axes = Axes(x_range=[-2, 2], y_range=[-1, 4], axis_config={"include_tip": False})
        curve = axes.plot(lambda x: x**2, color=WHITE)
        point = ValueTracker(-1)
        
        def get_point():
            x = point.get_value()
            return axes.c2p(x, x**2)
            
        dot = Dot(color=RED)
        dot.add_updater(lambda d: d.move_to(get_point()))
        
        self.place_in_area(axes, 'B3', 'F6', scale_factor=0.55)
        self.add(axes, curve, dot)
        
        self.place_at_grid(magnifier, 'C4', scale_factor=0.5)
        self.add(magnifier)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Imagine zooming into a curve to see slope.
        self.lecture[1].set_color(YELLOW)
        
        # Tangent line logic
        tangent = always_redraw(lambda: Line(
            start=axes.c2p(point.get_value() - 0.5, (point.get_value()**2) - 0.5 * (2 * point.get_value())),
            end=axes.c2p(point.get_value() + 0.5, (point.get_value()**2) + 0.5 * (2 * point.get_value())),
            color=YELLOW
        ))
        self.add(tangent)
        
        # Zooming in (visual simulation)
        self.play(point.animate.set_value(0), run_time=2)
        self.play(axes.animate.scale(3), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Slope defines the rate at that specific moment.
        self.lecture[2].set_color(GREEN)
        slope_label = MathTex("m = \\frac{dy}{dx}", color=GREEN)
        self.place_at_grid(slope_label, 'B4', scale_factor=0.9)
        self.add(slope_label)
        self.play(Flash(dot))
        self.wait(2)
