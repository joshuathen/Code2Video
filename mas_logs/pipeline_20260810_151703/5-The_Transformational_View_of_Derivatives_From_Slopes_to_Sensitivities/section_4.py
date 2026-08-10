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
        lecture_lines = ["Map each point to its steepness.", "The derivative reflects the curve's change.", "It visualizes the sensitivity function."]
        self.setup_layout("Visualizing the Derivative Function", lecture_lines)
        
        # Load assets
        graph_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/graph.svg")
        
        # Define functions and axes
        axes = Axes(x_range=[-2, 2, 1], y_range=[-1, 3, 1], axis_config={"include_tip": False}, x_length=3, y_length=2)
        f = lambda x: 0.5 * x**2 + 0.5
        f_prime = lambda x: x
        
        curve = axes.plot(f, color=WHITE)
        deriv = axes.plot(f_prime, color="#FF6600")
        
        graph_group = VGroup(axes, curve, deriv, graph_icon)
        self.place_in_area(graph_group, 'C3', 'F6', scale_factor=0.6)
        
        x_tracker = ValueTracker(-1.5)
        
        # Use simple shapes instead of complex dynamic always_redraw for performance
        dot = Dot(color=YELLOW)
        dot.add_updater(lambda d: d.move_to(axes.c2p(x_tracker.get_value(), f(x_tracker.get_value()))))
        self.place_at_grid(dot, 'D4', scale_factor=0.5)
        
        # Tangent line representation
        tangent = Line(start=LEFT*0.5, end=RIGHT*0.5, color=YELLOW)
        tangent.add_updater(lambda t: t.move_to(axes.c2p(x_tracker.get_value(), f(x_tracker.get_value()))))
        tangent.add_updater(lambda t: t.set_angle(np.arctan(f_prime(x_tracker.get_value()))))
        self.place_at_grid(tangent, 'E4', scale_factor=0.7)
        
        deriv_dot = Dot(color="#FF6600")
        deriv_dot.add_updater(lambda d: d.move_to(axes.c2p(x_tracker.get_value(), f_prime(x_tracker.get_value()))))

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.play(Create(axes), Create(curve), FadeIn(graph_icon))
        self.add(dot, tangent)
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF6600"))
        self.play(Create(deriv), run_time=2)
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00AAFF"))
        self.add(deriv_dot)
        self.play(x_tracker.animate.set_value(1.5), run_time=3, rate_func=linear)
        self.wait(1)
