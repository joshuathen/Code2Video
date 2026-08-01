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

class Section3Scene(Scene):
    def setup_layout(self):
        # Background and Title
        self.camera.background_color = "#000000"
        self.title = Text("Discrete Convolutions: Shifting & Blending", font_size=32, color=BLUE_B).to_edge(UP, buff=0.5)
        self.add(self.title)

        # Lecture Bullets on the Left
        bullet_points = [
            "- Discrete Convolution (f * g)",
            "- Sliding window mechanism",
            "- Element-wise multiplication",
            "- Summation of products",
            "- Resulting output sequence"
        ]
        
        self.lecture_group = VGroup(*[
            Text(point, font_size=20, color=WHITE) for point in bullet_points
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        
        self.lecture_group.to_edge(LEFT, buff=1.0).shift(DOWN * 0.5)
        self.add(self.lecture_group)

        # Define Visualization Area (Right side)
        self.viz_center = RIGHT * 3 + DOWN * 0.5

    def construct(self):
        self.setup_layout()

        # Input signals
        f = [1, 2, 3]
        g = [0.5, 1, 0.5]
        
        # Display Signal f - Using Text instead of MathTex to avoid LaTeX dependency
        f_label = Text("f[n] = [1, 2, 3]", font_size=24).move_to(self.viz_center + UP * 2)
        f_squares = VGroup(*[
            VGroup(Square(side_length=0.7, color=GREEN), Text(str(val), font_size=20))
            for val in f
        ]).arrange(RIGHT, buff=0.1).next_to(f_label, DOWN)

        # Display Signal g (Kernel) - Using Text instead of MathTex to avoid LaTeX dependency
        g_label = Text("g[n] = [0.5, 1, 0.5]", font_size=24).next_to(f_squares, DOWN, buff=0.5)
        g_squares = VGroup(*[
            VGroup(Square(side_length=0.7, color=RED), Text(str(val), font_size=20))
            for val in g
        ]).arrange(RIGHT, buff=0.1).next_to(g_label, DOWN)

        self.play(
            Write(f_label),
            Create(f_squares),
            run_time=1
        )
        self.wait(0.5)
        self.play(
            Write(g_label),
            Create(g_squares),
            run_time=1
        )

        # Animation: Sliding g over f (Simplified Visualization)
        sliding_g = g_squares.copy().set_opacity(0.7)
        indicator = Arrow(start=UP, end=DOWN, color=YELLOW, buff=0.1).scale(0.5)
        indicator.next_to(f_squares[0], UP)

        self.play(FadeIn(sliding_g), FadeIn(indicator))

        # Perform "Sliding" steps
        for i in range(len(f)):
            self.play(
                sliding_g.animate.move_to(f_squares[i].get_center() + DOWN * 0.8),
                indicator.animate.next_to(f_squares[i], UP),
                run_time=0.8
            )
            flash = f_squares[i][0].copy().set_color(YELLOW).set_stroke(width=5)
            self.play(Flash(f_squares[i], color=YELLOW), FadeOut(flash, run_time=0.3))

        # Conclusion text
        result_text = Text("Convolution yields blended output", font_size=22, color=YELLOW_B).to_edge(DOWN, buff=1.0)
        self.play(Write(result_text))
        
        self.wait(2)
