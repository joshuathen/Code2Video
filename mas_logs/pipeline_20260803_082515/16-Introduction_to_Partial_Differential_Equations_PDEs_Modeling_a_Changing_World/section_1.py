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
        # Data from storyboard
        title = "The Hook: From a Single Point to a Surface"
        lines = [
            "ODEs track a single value over time.",
            "Imagine a falling ball as a point.",
            "PDEs describe entire shapes or fields.",
            "Ripple effects spread in all directions.",
            "Everything changes simultaneously across the surface."
        ]
        
        self.setup_layout(title, lines)

        # Pre-set lecture line transparency to simulate "inactive" state
        for line in self.lecture:
            line.set_color(GRAY_E)

        # === Animation for Lecture Line 1 ===
        # "ODEs track a single value over time."
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        self.ode_dot = Dot(color=WHITE, radius=0.15)
        # Fix: Issue 21 - Move to B2
        self.place_at_grid(self.ode_dot, 'B2')
        
        # Movement logic: vertical oscillation to represent a changing value
        vt = ValueTracker(0)
        # Fix: Issue 21 - Use B2 center
        ode_center = self.grid['B2'].copy()
        self.ode_dot.add_updater(lambda d: d.move_to(ode_center + UP * 0.6 * np.sin(vt.get_value() * 3)))
        
        self.play(FadeIn(self.ode_dot))
        self.play(vt.animate.set_value(2), run_time=2, rate_func=linear)

        # === Animation for Lecture Line 2 ===
        # "Imagine a falling ball as a point."
        self.play(self.lecture[1].animate.set_color(WHITE))
        
        self.ode_label = Text("ODE", font_size=24, color=WHITE)
        # Fix: Issue 21 - Move to B3, scale 0.8
        self.place_at_grid(self.ode_label, 'B3', scale_factor=0.8)
        
        self.play(Write(self.ode_label))
        self.play(vt.animate.set_value(4), run_time=2, rate_func=linear)

        # === Animation for Lecture Line 3 ===
        # "PDEs describe entire shapes or fields."
        pde_color = "#55C1FF"
        self.play(self.lecture[2].animate.set_color(pde_color))
        
        # Mesh Grid (dots represent the "surface" or field sampled at points)
        self.dots = VGroup()
        for x in np.linspace(-1.3, 1.3, 8):
            for y in np.linspace(-1.3, 1.3, 8):
                dot = Dot(point=[x, y, 0], radius=0.05, color=pde_color)
                self.dots.add(dot)
        
        # Fix: Issue 23 - Move to C4-F6
        self.place_in_area(self.dots, 'C4', 'F6')
        
        # Label 'PDE'
        self.pde_label = Text("PDE", font_size=24, color=pde_color)
        # Fix: Issue 22 - Move to B5, scale 0.8
        self.place_at_grid(self.pde_label, 'B5', scale_factor=0.8)
        
        self.play(FadeIn(self.dots), Write(self.pde_label))

        # === Animation for Lecture Line 4 ===
        # "Ripple effects spread in all directions."
        self.play(self.lecture[3].animate.set_color(pde_color))
        
        pde_time = ValueTracker(0)
        dots_center = self.dots.get_center().copy()
        
        # Capture initial positions to avoid drift and maintain performance
        for dot in self.dots:
            dot.initial_pos = dot.get_center().copy()

        def update_dots(d):
            t = pde_time.get_value()
            for dot in d:
                dist = np.linalg.norm(dot.initial_pos - dots_center)
                # Ripple effect: vertical oscillation based on distance from center
                shift = 0.25 * np.sin(1.5 * PI * dist - 5 * t)
                dot.move_to(dot.initial_pos + UP * shift)
                # Also vary color slightly for visual depth
                dot.set_opacity(0.4 + 0.6 * np.cos(1.5 * PI * dist - 5 * t))

        self.dots.add_updater(update_dots)
        
        self.play(pde_time.animate.set_value(3), run_time=3, rate_func=linear)

        # === Animation for Lecture Line 5 ===
        # "Everything changes simultaneously across the surface."
        self.play(self.lecture[4].animate.set_color(pde_color))
        self.play(pde_time.animate.set_value(7), run_time=4, rate_func=linear)
        
        self.wait(2)
