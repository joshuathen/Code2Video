from manim import *
import numpy as np

class Section2Scene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # Background configuration
        self.camera.background_color = "#000000"
        
        # Title Setup
        self.title = Text(title_text, font_size=32, color=BLUE_A).to_edge(UP, buff=0.5)
        self.title_line = Line(LEFT, RIGHT, color=BLUE_E).scale(5).next_to(self.title, DOWN, buff=0.2)
        self.add(self.title, self.title_line)

        # Left-side lecture content
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        self.lecture.to_edge(LEFT, buff=0.5).shift(DOWN * 0.2)
        self.add(self.lecture)

        # Visualization Grid (Right Side)
        # We define a localized coordinate system for the right-hand panel
        self.grid_origin = RIGHT * 3 + DOWN * 0.5
        self.axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=5,
            y_length=5,
            axis_config={"include_tip": True, "color": GREY_B}
        ).move_to(self.grid_origin)
        
        self.add(self.axes)

    def construct(self):
        # Define specific content for the Riemann Zeta Function context
        lecture_lines = [
            "- The Euler Product Formula",
            "- Linking Primes to Infinite Series",
            "- Analytical Continuation",
            "- The Critical Strip (Re(s) = 1/2)",
            "- Connection to Prime Distribution"
        ]
        
        # Initialize Layout
        self.setup_layout("Section 2: The DNA of Prime Numbers", lecture_lines)
        
        # Mathematical Objects (The Zeta Formula)
        zeta_formula = MathTex(
            r"\zeta(s) = \sum_{n=1}^{\infty} \frac{1}{n^s}",
            font_size=36,
            color=YELLOW
        ).next_to(self.title_line, DOWN, buff=0.5).to_edge(RIGHT, buff=1.0)

        # Euler Product Visualization
        euler_formula = MathTex(
            r"= \prod_{p \text{ prime}} \frac{1}{1-p^{-s}}",
            font_size=32,
            color=GREEN
        ).next_to(zeta_formula, DOWN, buff=0.4)

        # Animation Sequence
        self.play(
            Write(zeta_formula),
            run_time=1.5
        )
        self.wait(0.5)
        
        self.play(
            FadeIn(euler_formula, shift=UP),
            run_time=1.5
        )

        # Illustrate a "Complex Point" on the axes
        s_point = Dot(self.axes.c2p(0.5, 2), color=RED)
        s_label = MathTex("s = 1/2 + it", font_size=24).next_to(s_point, UR, buff=0.1)
        
        # Path representing values of the function
        complex_path = ParametricFunction(
            lambda t: self.axes.c2p(np.cos(t) * np.exp(-0.1*t), np.sin(t) * np.exp(-0.1*t)),
            t_range=[0, 4*PI],
            color=GOLD
        )

        self.play(Create(s_point), Write(s_label))
        self.play(Create(complex_path), run_time=3)
        
        # Highlights
        rect = SurroundingRectangle(self.lecture[0], color=YELLOW, buff=0.1)
        self.play(Create(rect))
        self.wait(1)
        self.play(Uncreate(rect))

        self.wait(3)