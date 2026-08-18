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
        self.setup_layout("The Heat Equation: A Case Study", [
            "Heat moves through space over time.",
            "Temperature changes depend on local spatial curvature.",
            "Hot spots diffuse toward cooler areas."
        ])
        
        # Objects
        rod = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rod.svg")
        self.place_in_area(rod, 'A2', 'B5', scale_factor=0.8)
        
        # Initial heat spike (Gaussian-like)
        def get_heat_curve(t):
            x = np.linspace(-1.5, 1.5, 100)
            y = 1.0 * np.exp(-(x**2) / (0.1 + t))
            points = np.array([x, y, np.zeros_like(x)]).T
            # Offset to sit on rod
            points[:, 0] += rod.get_center()[0]
            points[:, 1] += rod.get_center()[1]
            return VMobject().set_points_smoothly(points).set_stroke(width=4)

        heat_curve = get_heat_curve(0).set_color(RED)
        heat_label = Text("Heat Spike", font_size=18, color=RED).next_to(heat_curve.get_top(), UP, buff=0.1)
        self.add(heat_curve, heat_label)

        laplacian_vec = Arrow(start=ORIGIN, end=UP*0.5, color=WHITE)
        laplacian_label = Text("Laplacian", font_size=16, color=WHITE)
        self.place_at_grid(laplacian_label, 'C4', scale_factor=0.6)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(RED))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.play(FadeIn(laplacian_vec.next_to(heat_curve.get_top(), UP)))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(ORANGE))
        
        # Animate spreading
        t_tracker = ValueTracker(0)
        
        def update_curve(m):
            t = t_tracker.get_value()
            new_curve = get_heat_curve(t).set_color(interpolate_color(RED, YELLOW, t/2))
            m.become(new_curve)
            
        heat_curve.add_updater(update_curve)
        
        # Update vector based on curvature (simple approximation)
        laplacian_vec.add_updater(lambda m: m.next_to(heat_curve.get_top(), UP))
        
        self.play(t_tracker.animate.set_value(2), run_time=3)
        self.play(laplacian_vec.animate.scale(0.2).set_opacity(0.3))
        
        heat_curve.remove_updater(update_curve)
        self.wait(2)
