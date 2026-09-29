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
            "Collisions equate to reflections in a wedge.",
            "Trajectory follows an arc in the wedge.",
            "The angle depends on the mass ratio.",
            "More bounces reveal more digits of Pi.",
            "[Asset: WedgeReflections] visualizes the geometric Pi."
        ]
        self.setup_layout("The Geometric Bridge to Pi", lecture_lines)
        
        wedge = Polygon(ORIGIN, [3, 1, 0], [3, -1, 0], color=BLUE)
        self.place_in_area(wedge, "A3", "D5", scale_factor=0.6)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(wedge))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        block = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg", color=RED)
        arc = Arc(radius=1.5, angle=PI/6, color=RED).shift(wedge.get_left())
        self.play(Create(arc), FadeIn(block.scale(0.2).next_to(arc, RIGHT)))
        self.lecture[1].set_color(YELLOW)

        # === Animation for Lecture Line 3 ===
        angle_label = MathTex(r"\\theta = \\arctan(\\sqrt{m/M})", font_size=24)
        self.place_at_grid(angle_label, "E4", scale_factor=0.7)
        self.play(Write(angle_label))
        self.lecture[2].set_color(YELLOW)

        # === Animation for Lecture Line 4 ===
        dots = VGroup(*[SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg", color=GREEN).scale(0.1).move_to(wedge.get_center() + 0.3*np.random.randn(3)) for _ in range(15)])
        self.play(LaggedStart(*[FadeIn(d) for d in dots]))
        self.lecture[3].set_color(YELLOW)

        # === Animation for Lecture Line 5 ===
        pi_digits = Text("3.14159265...", font_size=24, color=WHITE)
        self.place_at_grid(pi_digits, "F4", scale_factor=0.8)
        self.play(Write(pi_digits))
        self.lecture[4].set_color(YELLOW)
        self.wait(2)
