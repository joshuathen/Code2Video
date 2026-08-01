from manim import *
import numpy as np

# Use the provided TeachingScene base class without modification.
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
        # Data from shared state
        title_text = "The Two Golden Rules of PDFs"
        lecture_lines = [
            "First, we start with a potential PDF curve.",
            "We must move it above the horizontal axis.",
            "This ensures that every probability value is non-negative.",
            "Next, we fill the total area under this curve.",
            "This total area must always sum to exactly one."
        ]
        
        # Setup the layout
        self.setup_layout(title_text, lecture_lines)

        # Color definitions from storyboard
        RED_CURVE = "#e74c3c"
        GREEN_CHECK = "#2ecc71"
        BLUE_AREA = "#3498db"
        WHITE_TEXT = "#ffffff"

        # Setup Plotting Elements
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-0.5, 1.5, 0.5],
            x_length=5,
            y_length=3,
            axis_config={"include_tip": False, "include_numbers": False}
        )
        self.place_in_area(axes, "B1", "F6")

        # === Animation for Lecture Line 1 ===
        # "First, we start with a potential PDF curve."
        self.play(self.lecture[0].animate.set_color(RED_CURVE))
        
        # Curve starts partially below the axis
        pdf_curve = axes.plot(
            lambda x: 1.2 * np.exp(-0.5 * x**2) - 0.4,
            color=RED_CURVE
        )
        self.play(Create(axes), Create(pdf_curve))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "We must move it above the horizontal axis."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(RED_CURVE)
        )
        
        # Move curve above x-axis
        valid_curve = axes.plot(
            lambda x: 0.8 * np.exp(-0.5 * x**2),
            color=RED_CURVE
        )
        self.play(Transform(pdf_curve, valid_curve))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "This ensures that every probability value is non-negative."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(GREEN_CHECK)
        )
        
        rule1_math = MathTex("f(x) \\ge 0", color=GREEN_CHECK, font_size=36)
        self.place_at_grid(rule1_math, "A3")
        
        checkmark = Tex("\\checkmark", color=GREEN_CHECK, font_size=48)
        self.place_at_grid(checkmark, "A5", scale_factor=0.7) # Resolution for Issue 33
        
        self.play(Write(rule1_math), FadeIn(checkmark))
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        # "Next, we fill the total area under this curve."
        self.play(
            FadeOut(rule1_math), # Resolution for Issue 34
            FadeOut(checkmark),
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(BLUE_AREA)
        )
        
        # Shade the area under the curve
        area = axes.get_area(pdf_curve, x_range=[-3, 3], color=BLUE_AREA, opacity=0.5)
        self.play(FadeIn(area))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "This total area must always sum to exactly one."
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(WHITE_TEXT)
        )
        
        # Label the shaded area as 'Total Area = 1'
        # Resolution for Issue 32 (wide formula positioning)
        area_label = MathTex(r"\text{Total Area} = 1", color=WHITE_TEXT, font_size=36)
        self.place_in_area(area_label, "A2", "A4", scale_factor=0.9)
        
        self.play(Write(area_label))
        self.wait(3)
