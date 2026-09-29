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
        lecture_lines = [
            "Substitute roots into the generating function.",
            "Summing them extracts the target coefficients.",
            "Algebra solves complex counting problems instantly.",
            "We count configurations using complex rotation.",
            "The filter isolates the exact answer."
        ]
        self.setup_layout("Putting It All Together: An Application", lecture_lines)
        
        # Assets
        filter_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/filter.svg")
        result_box = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/box.svg")

        # Content
        coefficients = MathTex("a_0, a_1, a_2, a_3, a_4", font_size=36)
        extracted = MathTex("a_k", font_size=60, color="#FF9900")
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.place_at_grid(filter_icon, "B3", scale_factor=0.5)
        self.play(FadeIn(filter_icon))
        self.place_in_area(coefficients, "C4", "D5", scale_factor=0.75)
        self.play(Write(coefficients))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(BLUE))
        filter_dot = Dot(color=YELLOW).move_to(self.grid["B4"])
        self.play(filter_dot.animate.move_to(self.grid["D4"]), run_time=1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(BLUE))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(BLUE))
        rotation_circle = Circle(radius=0.5, color=BLUE_D).move_to(self.grid["E3"])
        self.play(Create(rotation_circle))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(BLUE))
        self.place_at_grid(result_box, "E5", scale_factor=0.5)
        self.play(FadeIn(result_box))
        self.place_at_grid(extracted, "E5", scale_factor=0.8)
        self.play(ReplacementTransform(coefficients, extracted))
        self.play(extracted.animate.scale(1.5))
        self.wait(1)
