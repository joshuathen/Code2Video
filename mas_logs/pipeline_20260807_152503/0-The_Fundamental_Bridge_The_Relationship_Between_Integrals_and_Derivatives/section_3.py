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
        self.setup_layout("The Accumulation Function", [
            "Define an accumulation function for area.",
            "The sweep visualizes area growth.",
            "Growth rate equals curve height."
        ])
        
        # Axes and Function
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 3, 1], axis_config={"include_tip": False})
        axes.set_color(WHITE)
        self.place_in_area(axes, "B2", "D5", scale_factor=0.55)
        self.add(axes)
        
        f = lambda t: 0.5 * (t - 1)**2 + 1
        curve = axes.plot(f, x_range=[0.5, 3.5], color="#FFFFFF")
        self.add(curve)
        
        # Asset: Scanner
        scanner = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/scanner.svg")
        self.place_at_grid(scanner, "A6", scale_factor=0.3)
        self.play(FadeIn(scanner))
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF00FF"))
        area_func_label = MathTex(r"A(x) = \int_a^x f(t)dt", font_size=30).set_color("#FF00FF")
        self.place_at_grid(area_func_label, "B6", scale_factor=0.75)
        self.play(Write(area_func_label))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        
        x_tracker = ValueTracker(1)
        sweep_line = always_redraw(lambda: axes.get_vertical_line(axes.c2p(x_tracker.get_value(), 0), line_func=Line, color="#00FFFF"))
        self.add(sweep_line)
        
        # Area fill
        area = always_redraw(lambda: axes.get_area(curve, x_range=[0.5, x_tracker.get_value()], color="#00FFFF", opacity=0.3))
        self.add(area)
        
        self.play(x_tracker.animate.set_value(3.0), run_time=3, rate_func=linear)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF0000"))
        
        # Asset: Hourglass
        hourglass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hourglass.svg")
        self.place_at_grid(hourglass, "F6", scale_factor=0.3)
        self.play(FadeIn(hourglass))
        
        height_line = always_redraw(lambda: Line(axes.c2p(x_tracker.get_value(), 0), axes.c2p(x_tracker.get_value(), f(x_tracker.get_value())), color="#FF0000", stroke_width=4))
        self.add(height_line)
        
        rate_label = MathTex(r"A'(x) = f(x)", font_size=30).set_color("#FFFF00")
        self.place_at_grid(rate_label, "D6", scale_factor=0.75)
        self.play(Write(rate_label))
        
        self.play(x_tracker.animate.set_value(1.0), run_time=2, rate_func=linear)
        self.wait(1)
