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

class Section6Scene(TeachingScene):
    def construct(self):
        # Setup layout
        title_text = "Real-World Application: The Growing Bubble"
        lecture_lines = [
            "These techniques are essential for solving related rates problems.",
            "Imagine an expanding bubble where volume and radius change.",
            "Implicit differentiation links these rates of change together."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        related_rates_title = Text("Related Rates", font_size=24, color=WHITE)
        self.place_in_area(related_rates_title, "A2", "A5")
        
        # Creating a translucent sphere (using Circle for 2D representation)
        bubble = Circle(radius=0.7, color="#ADD8E6", fill_opacity=0.4, stroke_width=2)
        # Place in a central grid area
        self.place_in_area(bubble, "B2", "D5")
        
        self.play(
            FadeIn(related_rates_title),
            FadeIn(bubble)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Define arrows for rates of change
        # dr/dt arrow: radius direction
        dr_dt_arrow = Arrow(
            start=bubble.get_center(),
            end=bubble.get_center() + RIGHT * 1.2,
            color=YELLOW,
            buff=0
        )
        # Using Text instead of MathTex to avoid LaTeX dependency error
        dr_dt_label = Text("dr/dt", font_size=24, color=YELLOW)
        self.place_at_grid(dr_dt_label, "C6")
        
        # dV/dt arrow: pointing outward from surface
        dv_dt_arrow = Arrow(
            start=bubble.get_top(),
            end=bubble.get_top() + UP * 0.6,
            color=YELLOW,
            buff=0
        )
        # Using Text instead of MathTex to avoid LaTeX dependency error
        dv_dt_label = Text("dV/dt", font_size=24, color=YELLOW)
        self.place_at_grid(dv_dt_label, "B3")

        self.play(
            bubble.animate.scale(1.5),
            FadeIn(dr_dt_arrow),
            FadeIn(dr_dt_label),
            FadeIn(dv_dt_arrow),
            FadeIn(dv_dt_label),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Volume Equation - Using Text with Unicode characters for exponents
        volume_eq = Text("V = 4/3 πr³", font_size=28, color=WHITE)
        self.place_in_area(volume_eq, "E2", "E5")
        
        # Related Rates Equation (Derivative)
        derivative_eq = Text("dV/dt = 4πr² dr/dt", font_size=28, color=WHITE)
        self.place_in_area(derivative_eq, "F2", "F5")
        
        self.play(Write(volume_eq))
        self.wait(1)
        self.play(ReplacementTransform(volume_eq.copy(), derivative_eq))
        self.wait(2)