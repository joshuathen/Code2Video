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
        # Setup title and lecture lines
        title = "The Two Pillars of a Valid PDF"
        lines = [
            "First, the curve never drops below zero.",
            "Negative probability is physically impossible in our world.",
            "Second, the total area must equal exactly one."
        ]
        self.setup_layout(title, lines)

        # Hexadecimal colors (L008)
        RED_COLOR = "#FF0000"
        GREEN_COLOR = "#00FF00"
        PURPLE_COLOR = "#E6E6FA"
        WHITE_COLOR = "#FFFFFF"

        # Axes setup
        axes = Axes(
            x_range=[-2, 2, 1],
            y_range=[-1, 2, 1],
            x_length=4.5,
            y_length=4,
            axis_config={"include_tip": False, "color": WHITE_COLOR}
        )
        # Position visuals in the grid area B2-F6 (L002)
        self.place_in_area(axes, "B2", "F6", scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        # Line 1: First, the curve never drops below zero.
        self.play(self.lecture[0].animate.set_color(RED_COLOR))
        
        # Initial dipping curve: f(x) = x^2 - 0.5
        dipping_curve = axes.plot(lambda x: x**2 - 0.5, x_range=[-1.5, 1.5], color=WHITE_COLOR)
        
        # Highlight negative part in red (#FF0000)
        negative_part = axes.plot(lambda x: x**2 - 0.5, x_range=[-0.707, 0.707], color=RED_COLOR)
        
        # Add 'X' label - Fixed position to avoid overlap with axis (Issue 32)
        cross_mark = Text("X", color=RED_COLOR)
        self.place_at_grid(cross_mark, "E4", scale_factor=1.2)

        self.play(Create(axes))
        self.play(Create(dipping_curve))
        self.wait(0.5)
        self.play(Create(negative_part), Write(cross_mark))
        self.play(Indicate(cross_mark, color=RED_COLOR)) # L004
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # Line 2: Negative probability is physically impossible in our world.
        self.play(
            self.lecture[0].animate.set_color(WHITE_COLOR),
            self.lecture[1].animate.set_color(GREEN_COLOR)
        )
        
        # Correct the curve: f(x) = e^(-x^2) stays above the axis
        valid_curve = axes.plot(lambda x: np.exp(-x**2), x_range=[-2, 2], color=GREEN_COLOR)
        
        # Morph dipping curve into corrected green curve
        self.play(
            FadeOut(negative_part),
            FadeOut(cross_mark),
            ReplacementTransform(dipping_curve, valid_curve)
        )
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # Line 3: Second, the total area must equal exactly one.
        self.play(
            self.lecture[1].animate.set_color(WHITE_COLOR),
            self.lecture[2].animate.set_color(PURPLE_COLOR)
        )
        
        # Shade the entire area under the curve in soft purple (#E6E6FA)
        area = axes.get_area(valid_curve, x_range=[-2, 2], color=PURPLE_COLOR, opacity=0.4)
        
        # Display the total area equation: ∫ f(x) dx = 1 - Fixed position (Issue 33)
        # Using Text for robustness if MathTex environment is tricky, 
        # but the prompt specifically asked for MathTex/Tex-like notation in storyboard.
        # Let's use MathTex and hope for the best, or fallback if needed.
        try:
            equation = MathTex(r"\int_{-\infty}^{\infty} f(x) dx = 1", color=WHITE_COLOR)
        except:
            equation = Text("∫ f(x) dx = 1", color=WHITE_COLOR)
            
        self.place_in_area(equation, "A4", "A6", scale_factor=0.8)
        
        # Add a '100%' label to signify total certainty - Fixed position (Issue 34)
        label_100 = Text("100%", font_size=24, color=WHITE_COLOR)
        self.place_at_grid(label_100, "C4", scale_factor=1.0)
        
        self.play(FadeIn(area))
        self.wait(0.5)
        self.play(Write(equation))
        self.play(Write(label_100))
        self.play(Indicate(label_100, color=WHITE_COLOR))
        
        self.wait(2.0)
