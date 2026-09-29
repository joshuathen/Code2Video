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
        self.setup_layout("Solving the Integral", [
            "Integrate over radius r and angle θ.",
            "The 2π factor appears from integration.",
            "This 2π is our hidden π source."
        ])
        
        # Setup objects
        integral_expr = MathTex(r"I^2 = \int_0^{2\pi} d\theta \int_0^\infty r e^{-r^2} dr")
        self.place_in_area(integral_expr, 'B2', 'B5', scale_factor=0.9)

        # Assets
        radar_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/radar.svg")

        # === Animation for Lecture Line 1 ===
        self.play(Write(integral_expr), run_time=1)
        self.play(self.lecture[0].animate.set_color("#3498DB"))

        # === Animation for Lecture Line 2 ===
        # Radar Sweep simulation
        circle = Circle(radius=1.5, color=BLUE)
        self.place_at_grid(circle, 'D2', scale_factor=0.5)
        radar_icon.move_to(circle.get_center())
        radar_icon.scale(0.5)
        
        beam = Line(start=circle.get_center(), end=circle.get_center() + np.array([0.75, 0, 0]), color=YELLOW)
        
        self.add(circle, radar_icon, beam)
        self.play(Rotate(beam, angle=2*PI, about_point=circle.get_center()), run_time=2)
        
        self.play(self.lecture[1].animate.set_color("#2ECC71"))

        # === Animation for Lecture Line 3 ===
        final_val = MathTex(r"= 2\pi \cdot \frac{1}{2} = \pi")
        self.place_at_grid(final_val, 'D5', scale_factor=1.1)
        
        # Add radar icon again for visual theme
        radar_icon_2 = radar_icon.copy()
        self.place_at_grid(radar_icon_2, 'E5', scale_factor=0.3)
        
        self.play(Write(final_val), FadeIn(radar_icon_2))
        self.play(self.lecture[2].animate.set_color("#F1C40F"))
        
        self.wait(2)
