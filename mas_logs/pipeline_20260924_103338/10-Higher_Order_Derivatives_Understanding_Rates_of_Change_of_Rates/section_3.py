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
            "Higher orders follow this same hierarchical pattern.",
            "The third derivative often represents sudden \"jerk\".",
            "It measures rapid change in acceleration."
        ]
        self.setup_layout("Higher Orders (n-th derivatives)", lecture_lines)
        
        colors = [BLUE_B, YELLOW_B, RED_B]
        
        # === Animation for Lecture Line 1 ===
        seq = VGroup(
            MathTex("f'(x)"),
            MathTex("f''(x)"),
            MathTex("f'''(x)")
        ).arrange(RIGHT, buff=0.5)
        # Fix: using place_in_area as requested
        self.place_in_area(seq, 'A4', 'C6', scale_factor=1.0)
        
        self.play(Write(seq), self.lecture[0].animate.set_color(colors[0]))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        jerk_label = MathTex("f^{(n)}(x) = \\frac{d^n}{dx^n}f(x)")
        # Fix: using place_at_grid
        self.place_at_grid(jerk_label, 'D4', scale_factor=0.9)
        
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/car.png
        car = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/car.png")
        self.place_at_grid(car, 'F4', scale_factor=0.5)
        
        self.play(FadeIn(jerk_label), FadeIn(car), self.lecture[1].animate.set_color(colors[1]))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        all_formulas_group = VGroup(seq, jerk_label)
        # Fix: using place_in_area for combined group
        self.place_in_area(all_formulas_group, 'B4', 'E6', scale_factor=0.85)
        
        # Illustrate change: car motion along a curve
        curve = CubicBezier(np.array([1,0,0]), np.array([2,1,0]), np.array([3,0,0]), np.array([4,1,0]))
        self.play(Create(curve), self.lecture[2].animate.set_color(colors[2]))
        self.play(MoveAlongPath(car, curve), run_time=2)
        self.wait(2)
