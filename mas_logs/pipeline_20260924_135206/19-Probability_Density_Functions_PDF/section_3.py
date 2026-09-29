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
        lecture_lines = [
            "Probability is defined as area under the curve.",
            "Integrate the PDF between two points for probability.",
            "Shaded regions represent specific interval likelihoods."
        ]
        self.setup_layout("Defining Probability as Area", lecture_lines)
        
        # Define Axes
        axes = Axes(
            x_length=4,
            y_length=3,
            x_range=[0, 6, 1],
            y_range=[0, 1.5, 0.5],
            axis_config={"include_numbers": False}
        )
        self.place_in_area(axes, 'B3', 'E6', scale_factor=0.55)
        
        # PDF curve (Normal-like)
        pdf_curve = axes.plot(lambda x: 1.2 * np.exp(-(x - 3)**2 / 2), color=WHITE)
        
        # Interval points
        a, b = 2, 4
        area = axes.get_area(pdf_curve, x_range=[a, b], color="#00FF00", opacity=0.5)
        
        integral_label = MathTex(r"\int_{a}^{b} f(x)dx", color="#FFD700").scale(0.8)
        self.place_at_grid(integral_label, 'A4', scale_factor=0.9)
        
        # Probability Gauge (SVG)
        gauge_bar = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/gauge.svg")
        self.place_in_area(gauge_bar, 'E3', 'E5', scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.play(Create(axes), Create(pdf_curve))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFD700"))
        self.play(Create(area), Write(integral_label))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        self.play(FadeIn(gauge_bar))
        self.wait(2)
