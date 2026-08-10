from manim import *
import os

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
            "The Fundamental Theorem connects these two operations.",
            "They are inverse processes acting on functions.",
            "A scanner reveals how one rebuilds the other.",
            "Derivative deconstructs growth into instantaneous rates.",
            "Integral reconstructs the total from those rates."
        ]
        self.setup_layout("The Fundamental Theorem (Visualizing the Link)", lecture_lines)
        
        # Define objects
        f_x = MathTex("F(x)", color=WHITE)
        integral_f = MathTex(r"\\int_{a}^{x} f(t) dt", color=WHITE)
        link_arrow = Arrow(LEFT, RIGHT, color="#FF3366")
        equivalence = MathTex(r"\\frac{d}{dx} \\int_{a}^{x} f(t) dt = f(x)", color=WHITE)
        scanner = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/scanner.svg")

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        self.place_in_area(f_x, 'B3', 'B4', scale_factor=0.9)
        self.play(Write(f_x))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color("#FFD700"))
        self.place_in_area(integral_f, 'C3', 'C5', scale_factor=1.1)
        self.play(Write(integral_f))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color("#FFD700"))
        self.place_at_grid(link_arrow, 'D4', scale_factor=0.8)
        self.play(GrowArrow(link_arrow))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[2].animate.set_color(WHITE), self.lecture[3].animate.set_color("#FFD700"))
        # Using scanner asset to highlight x
        self.place_in_area(scanner, 'B5', 'C6', scale_factor=0.5)
        self.play(FadeIn(scanner))
        self.play(scanner.animate.move_to(f_x.get_center()), scanner.animate.move_to(integral_f.get_center()))
        self.play(f_x.animate.set_color(YELLOW), integral_f.animate.set_color(YELLOW))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[3].animate.set_color(WHITE), self.lecture[4].animate.set_color("#FFD700"))
        self.play(FadeOut(f_x), FadeOut(integral_f), FadeOut(link_arrow), FadeOut(scanner))
        self.place_in_area(equivalence, "B3", "E5", scale_factor=1.2)
        self.play(Write(equivalence))
        self.wait(2)
