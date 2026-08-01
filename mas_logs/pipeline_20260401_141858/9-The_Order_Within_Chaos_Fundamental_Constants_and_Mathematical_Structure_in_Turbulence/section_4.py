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
        title = "The -5/3 Power Law: Kolmogorov’s Universal Constant"
        lines = [
            "We map energy distribution using a log-log plot.",
            "Turbulent data points initially appear messy and random.",
            "Remarkably, they collapse onto a specific -5/3 slope.",
            "This power law defines the universal mathematical structure.",
            "Kolmogorov's law governs turbulence across the entire universe."
        ]
        self.setup_layout(title, lines)

        # Colors
        BLUE_DATA = "#58C4DD"
        GOLD_TERM = "#F8E71C"
        GREEN_TAG = "#83C167"
        AXES_COLOR = WHITE

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE_DATA))
        
        # Log-log axes
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 5, 1],
            axis_config={"color": AXES_COLOR, "include_tip": True},
            x_length=4.5,
            y_length=3.5,
        )
        # Using Text for robustness against LaTeX errors
        x_label = Text("k", font_size=20, slant=ITALIC).next_to(axes.x_axis, RIGHT, buff=0.1)
        y_label = Text("E(k)", font_size=20, slant=ITALIC).next_to(axes.y_axis, UP, buff=0.1)
        plot_group = VGroup(axes, x_label, y_label)
        
        # Issue 38: Vertical compression and unused space fix
        self.place_in_area(plot_group, "B2", "F5", scale_factor=0.9)
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(BLUE_DATA))
        
        # Scatter messy blue data points
        np.random.seed(42)
        points = VGroup()
        for _ in range(40):
            x = np.random.uniform(0.5, 4.5)
            y = np.random.uniform(0.5, 4.5)
            dot = Dot(axes.c2p(x, y), radius=0.04, color=BLUE_DATA)
            points.add(dot)
        
        self.play(FadeIn(points, lag_ratio=0.1))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(GOLD_TERM))
        
        # Points collapse into a straight line with slope -5/3
        # In log-log space, a power law E(k) = k^(-5/3) is a line with slope -5/3
        # y = -5/3 * x + intercept
        spectral_line = Line(axes.c2p(0.5, 4.5), axes.c2p(3.5, -0.5), color=GOLD_TERM, stroke_width=4)
        
        animations = []
        for i, dot in enumerate(points):
            # Generate target points on the line
            x_val = np.random.uniform(0.7, 3.3)
            y_val = -1.66 * x_val + 5.3  # Visual approximation of -5/3 slope
            animations.append(dot.animate.move_to(axes.c2p(x_val, y_val)).set_color(GOLD_TERM))
            
        self.play(*animations)
        self.play(Create(spectral_line))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[2].animate.set_color(WHITE), self.lecture[3].animate.set_color(GOLD_TERM))
        
        # Using Text parts for specific highlighting
        formula_base = Text("E(k) = C ε^(2/3) ", font_size=32)
        formula_exp = Text("k^(-5/3)", font_size=32, color=GOLD_TERM)
        formula = VGroup(formula_base, formula_exp).arrange(RIGHT, buff=0.1)
        
        # Issue 36: Formula positioning fix
        self.place_in_area(formula, 'A1', 'A6', scale_factor=0.7)
        
        self.play(Write(formula))
        self.play(Indicate(formula_exp, color=GOLD_TERM, scale_factor=1.2))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[3].animate.set_color(WHITE), self.lecture[4].animate.set_color(GREEN_TAG))
        
        # Issue 26: Integrate universe icon asset
        universe_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/universe.svg")
        universe_icon.set_color(GREEN_TAG)
        
        universal_tag = Text("UNIVERSAL", font_size=24, color=GREEN_TAG, weight=BOLD)
        
        # Combine into a group for easier flashing
        universal_group = VGroup(universal_tag, universe_icon).arrange(RIGHT, buff=0.2)
        
        # Issue 37: Universal tag positioning fix
        self.place_at_grid(universal_group, 'B6', scale_factor=0.6)
        
        # Flashing animation
        for _ in range(3):
            self.play(FadeIn(universal_group), run_time=0.3)
            self.play(FadeOut(universal_group), run_time=0.3)
        self.play(FadeIn(universal_group))
        
        self.wait(2)
