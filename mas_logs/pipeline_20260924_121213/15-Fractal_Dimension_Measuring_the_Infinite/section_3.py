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
        lecture_lines = ["Koch snowflakes begin with a straight line.",
                         "Replace segments with four, one-third length.",
                         "Iteration repeats infinitely for self-similarity.",
                         "Calculation yields: D equals log 4 over log 3.",
                         "Result is approximately one point two six."]
        
        self.setup_layout("Calculating the Fractal Dimension: The Koch Snowflake", lecture_lines)

        # === Animation for Lecture Line 1 ===
        snowflake_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/snowflake.svg", color="#00FF00")
        self.place_at_grid(snowflake_icon, 'B4', scale_factor=0.6)
        self.play(Create(snowflake_icon))
        self.lecture[0].set_color("#00FF00")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Representing segment replacement
        segments = VGroup(*[Line(color="#FF00FF") for _ in range(4)])
        self.place_at_grid(segments, 'C4', scale_factor=0.6)
        self.play(ReplacementTransform(snowflake_icon, segments))
        self.lecture[1].set_color("#FF00FF")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(segments.animate.set_color("#00FFFF"))
        self.lecture[2].set_color("#00FFFF")
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        formula = MathTex(r"D = \frac{\log(4)}{\log(3)}", color="#FFFF00")
        self.place_at_grid(formula, 'D4', scale_factor=0.8)
        self.play(Write(formula))
        self.lecture[3].set_color("#FFFF00")
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        result = Text(r"$\approx 1.26$", color="#FF8800")
        self.place_at_grid(result, 'E4', scale_factor=0.6)
        
        final_snowflake = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/snowflake.svg", color="#00FF00")
        self.place_at_grid(final_snowflake, 'B5', scale_factor=0.4)
        
        self.play(FadeIn(result), FadeIn(final_snowflake))
        self.lecture[4].set_color("#FF8800")
        self.wait(2)
