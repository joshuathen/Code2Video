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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Roots of Unity Filter", [
            "Complex numbers can filter specific coefficients.",
            "Use roots of unity to identify indices.",
            "The filter uses (1/k) sum f(omega^j).",
            "Roots rotate around the unit circle.",
            "Phases cancel terms not divisible by k."
        ])

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        circle = Circle(radius=1.5, color="#FFFFFF")
        # Placing compass and circle at C4
        self.place_at_grid(compass, 'C4', scale_factor=0.5)
        self.place_at_grid(circle, 'C4', scale_factor=0.7)
        self.play(FadeIn(compass), Create(circle))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFD700")
        k = 6
        roots = [np.exp(2j * PI * j / k) for j in range(k)]
        dots = VGroup(*[Dot(point=np.array([r.real, r.imag, 0]) * 1.5 + circle.get_center(), color="#FFD700") for r in roots])
        self.play(FadeIn(dots))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00CED1")
        formula = MathTex(r"\\frac{1}{k} \\sum f(\\omega^j)", color="#00CED1")
        self.place_in_area(formula, 'E4', 'E5', scale_factor=0.75)
        self.play(Write(formula))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFD700")
        polygon = Polygon(*[d.get_center() for d in dots], color="#00CED1")
        self.play(Create(polygon))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFFFFF")
        self.wait(1)
