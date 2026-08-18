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
        lecture_lines = [
            "Algebra lets us extend spheres to any dimension n.",
            "The formula is the sum of n squared coordinates.",
            "We call these higher-dimensional shapes hyperspheres."
        ]
        self.setup_layout("The Leap to N-Dimensions", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Display the 3D sphere equation x1^2 + x2^2 + x3^2 = 1 in #FFFFFF.
        self.lecture[0].set_color("#FFFFFF")
        eq1 = MathTex("x_1^2 + x_2^2 + x_3^2 = 1", color="#FFFFFF")
        self.place_in_area(eq1, 'B2', 'B5', scale_factor=1.2)
        
        self.play(Write(eq1))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Animate the equation growing into x1^2 + x2^2 +... + xn^2 = 1 as terms are added in #FF00FF.
        self.lecture[0].set_color(GRAY)
        self.lecture[1].set_color("#FF00FF")
        
        eq2 = MathTex("x_1^2 + x_2^2 + \\dots + x_n^2 = 1", color="#FF00FF")
        self.place_in_area(eq2, 'B2', 'B5', scale_factor=1.2)
        
        self.play(Transform(eq1, eq2))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Visualize a shimmering, multi-layered pulse in #A020F0 to represent the conceptual Hypersphere.
        self.lecture[1].set_color(GRAY)
        self.lecture[2].set_color("#A020F0")
        
        # Creating a pulse effect with circles
        pulse_group = VGroup()
        base_radii = [0.4, 0.7, 1.0, 1.3, 1.6]
        for i, r in enumerate(base_radii):
            circle = Circle(radius=r, color="#A020F0", stroke_opacity=0.8 - i*0.12)
            circle.initial_radius = r
            pulse_group.add(circle)
        
        self.place_in_area(pulse_group, 'D1', 'F6', scale_factor=0.8)
        
        # ValueTracker for the shimmering pulse effect
        pulse_tracker = ValueTracker(0)
        
        # Apply updaters for shimmering effect (scaling and opacity)
        for i, circle in enumerate(pulse_group):
            circle.add_updater(
                lambda m, i=i: m.set_stroke(
                    opacity=(0.4 + 0.3 * np.sin(pulse_tracker.get_value() * 3 + i))
                ).scale_to_fit_width(
                    2 * m.initial_radius * (1 + 0.05 * np.sin(pulse_tracker.get_value() * 5 + i))
                )
            )

        self.play(Create(pulse_group))
        self.play(pulse_tracker.animate.set_value(10), run_time=5, rate_func=linear)
        self.wait(2)
        
        # Reset colors
        for line in self.lecture:
            line.set_color(WHITE)
        self.wait(1)
