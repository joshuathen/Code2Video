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
        # Initial Setup
        title_str = "Conclusion: The Unsolved Square Peg Problem"
        lines = [
            "Every smooth loop contains an inscribed rectangle.",
            "Finding a square remains a famous mystery.",
            "Topology reveals hidden structures within simple shapes."
        ]
        self.setup_layout(title_str, lines)

        # Colors
        GOLD = "#FFD700"
        RED = "#FF0000"
        WHITE = "#FFFFFF"
        YELLOW = "#FFFF00"
        CYAN = "#00FFFF"
        PINK_COLOR = "#FFC0CB"

        # === Animation for Lecture Line 1 ===
        # Line 1: Every smooth loop contains an inscribed rectangle.
        self.play(self.lecture[0].animate.set_color(YELLOW))

        # Shape 1: Star (using Asset)
        star_svg = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/star.svg").set_color(BLUE_B)
        star_sq = Square(side_length=0.45, color=GOLD).rotate(PI/4)
        star_group = VGroup(star_svg, star_sq)
        self.place_at_grid(star_group, "B3", scale_factor=0.6) # Shifted 2->3 per Issue 47

        # Shape 2: Heart
        heart = ParametricFunction(
            lambda t: np.array([
                16 * np.sin(t)**3,
                13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t),
                0
            ]),
            t_range=[0, TAU],
            color=PINK_COLOR
        ).scale(0.04)
        heart_sq = Square(side_length=0.45, color=GOLD)
        heart_group = VGroup(heart, heart_sq)
        self.place_at_grid(heart_group, "B6", scale_factor=1.0) # Shifted 5->6 per Issue 48

        # Shape 3: Ellipse
        ellipse = Ellipse(width=1.8, height=1.0, color=GREEN_B)
        ellipse_sq = Square(side_length=0.65, color=GOLD)
        ellipse_group = VGroup(ellipse, ellipse_sq)
        self.place_at_grid(ellipse_group, "D3", scale_factor=0.6) # Shifted E->D (Issue 46), 2->3 (Issue 47)

        # Shape 4: Blob
        blob = Circle(radius=0.7, color=PURPLE_B).stretch(1.2, 0).stretch(0.8, 1)
        blob_sq = Square(side_length=0.55, color=GOLD).rotate(PI/8)
        blob_group = VGroup(blob, blob_sq)
        self.place_at_grid(blob_group, "D6", scale_factor=0.6) # Shifted E->D (Issue 46), 5->6 (Issue 48)

        grid_shapes = VGroup(star_group, heart_group, ellipse_group, blob_group)
        self.play(FadeIn(grid_shapes))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line 2: Finding a square remains a famous mystery.
        self.play(
            self.lecture[1].animate.set_color(RED),
            self.lecture[0].animate.set_color(WHITE)
        )

        # Create a fractal-like complex curve
        # We'll use a jagged loop to represent complexity
        def complex_loop(t):
            r = 1.0 + 0.15 * np.sin(10 * t) + 0.1 * np.cos(25 * t)
            return np.array([r * np.cos(t), r * np.sin(t), 0])

        fractal_curve = ParametricFunction(complex_loop, t_range=[0, TAU], color=CYAN)
        self.place_in_area(fractal_curve, "B3", "D6", scale_factor=1.2)
        
        question_mark = Text("?", font_size=72, color=RED)
        self.place_in_area(question_mark, "B3", "D6", scale_factor=1.0)

        self.play(
            ReplacementTransform(grid_shapes, fractal_curve),
            FadeIn(question_mark)
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Line 3: Topology reveals hidden structures within simple shapes.
        self.play(
            self.lecture[2].animate.set_color(WHITE), # Line 3 text color
            self.lecture[1].animate.set_color(WHITE)
        )
        
        conjecture_text = Text("Toeplitz's Conjecture", font_size=32, color=WHITE)
        self.place_in_area(conjecture_text, "A3", "F6", scale_factor=1.0)

        self.play(
            FadeOut(fractal_curve),
            FadeOut(question_mark),
            FadeIn(conjecture_text)
        )
        self.wait(3)
