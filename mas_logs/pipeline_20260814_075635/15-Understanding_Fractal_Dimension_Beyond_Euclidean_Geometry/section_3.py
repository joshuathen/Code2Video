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
        lecture_text = [
            "The fractal dimension formula is log(N) over log(r).",
            "D measures how shapes fill space efficiently.",
            "Koch snowflakes have a dimension of 1.26.",
            "This confirms D is not always an integer.",
            "Fractals are infinitely complex and detailed."
        ]
        self.setup_layout("Defining Fractal Dimension (Hausdorff Dimension)", lecture_text)
        
        # === Animation for Lecture Line 1 ===
        # Display Koch curve generator
        snowflake = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/snowflake.svg", color=WHITE)
        self.place_at_grid(snowflake, 'B3', scale_factor=0.5)
        self.play(FadeIn(snowflake))
        
        formula = MathTex(r"D = \frac{\log(N)}{\log(r)}", color=YELLOW)
        self.place_at_grid(formula, 'B2', scale_factor=1.5)
        self.play(Write(formula))
        self.play(self.lecture[0].animate.set_color(YELLOW))

        # === Animation for Lecture Line 2 ===
        label_N = MathTex(r"N=4", color="#3357FF")
        label_r = MathTex(r"r=3", color="#3357FF")
        self.place_at_grid(label_N, 'D3', scale_factor=1.0)
        self.place_at_grid(label_r, 'D4', scale_factor=1.0)
        self.play(FadeIn(label_N), FadeIn(label_r))
        self.play(self.lecture[1].animate.set_color("#3357FF"))

        # === Animation for Lecture Line 3 ===
        new_formula = MathTex(r"D = \frac{\log(4)}{\log(3)} \approx 1.26", color="#00FF00")
        self.place_in_area(new_formula, 'B2', 'C5', scale_factor=1.1)
        self.play(ReplacementTransform(formula, new_formula))
        self.play(self.lecture[2].animate.set_color("#00FF00"))

        # === Animation for Lecture Line 4 ===
        highlight = SurroundingRectangle(new_formula[10:], color=RED)
        self.play(Create(highlight))
        self.play(self.lecture[3].animate.set_color(RED))

        # === Animation for Lecture Line 5 ===
        # Compare 1.26 to Euclidean 1D line using snowflake asset icon again as requested in storyboard
        snowflake_line = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/snowflake.svg", color=WHITE)
        self.place_at_grid(snowflake_line, 'C6', scale_factor=0.3)
        self.play(FadeIn(snowflake_line))
        
        final_text = Text("Self-Similarity", font_size=36, color=WHITE)
        self.place_at_grid(final_text, 'D5', scale_factor=0.9)
        self.play(FadeIn(final_text))
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(2)
