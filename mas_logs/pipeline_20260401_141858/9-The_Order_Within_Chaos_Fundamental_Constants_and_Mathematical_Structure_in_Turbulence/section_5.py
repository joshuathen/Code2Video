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
        # Teaching Content
        title = "Dissipation and the Kolmogorov Microscale"
        lecture_lines = [
            "Energy eventually reaches the smallest Kolmogorov scales.",
            "Viscous forces turn kinetic energy into internal heat.",
            "Motion ceases as friction dissipates the remaining energy."
        ]
        
        self.setup_layout(title, lecture_lines)

        blue_color = "#58C4DD"
        red_color = "#FC6255"

        # === Animation for Lecture Line 1 ===
        # Show small blue (#58C4DD) eddies spinning at the microscale.
        self.play(self.lecture[0].animate.set_color(blue_color))
        
        def create_eddy(radius, color):
            return Arc(radius=radius, start_angle=0, angle=3*PI/2, color=color).add_tip(tip_length=radius*0.4)

        # Create a cluster of blue eddies (swirls)
        swirls = VGroup()
        for i in range(6):
            r = 0.1 + (i % 3) * 0.05
            e = create_eddy(r, blue_color)
            swirls.add(e)
        
        swirls.arrange_in_grid(rows=3, cols=2, buff=0.6)
        # Issue 40 Fix: self.place_in_area(swirls, 'B1', 'E4', scale_factor=0.7)
        self.place_in_area(swirls, 'B1', 'E4', scale_factor=0.7)

        # Kolmogorov formula for visual context
        formula = Text("η = (ν³ / ε)¹/⁴", font_size=32)
        # Issue 39 Fix: self.place_in_area(formula, 'A2', 'A5', scale_factor=0.8)
        self.place_in_area(formula, 'A2', 'A5', scale_factor=0.8)

        # Scale label η
        eta_label = Text("η", font_size=24)
        # Issue 41 Fix: self.place_at_grid(eta_label, 'D5', scale_factor=0.8)
        self.place_at_grid(eta_label, 'D5', scale_factor=0.8)

        self.play(Create(swirls), Write(formula), Write(eta_label))
        self.play(*(Rotate(e, angle=4*PI, rate_func=linear) for e in swirls), run_time=3)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # The eddies turn red (#FC6255) to represent viscous heating.
        self.play(self.lecture[1].animate.set_color(red_color))
        self.play(swirls.animate.set_color(red_color))
        self.play(*(Rotate(e, angle=2*PI, rate_func=linear) for e in swirls), run_time=2)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # The red eddies fade away as motion stops completely.
        self.play(self.lecture[2].animate.set_color(GRAY))
        self.play(
            FadeOut(swirls),
            FadeOut(formula),
            FadeOut(eta_label),
            run_time=2
        )
        self.wait(1)
