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
            "We extend the function via continuation.",
            "The critical strip holds chaotic behavior.",
            "Zeros here define the Riemann Hypothesis.",
            "It is like unfolding a map.",
            "We search for these hidden zeros."
        ]
        self.setup_layout("Analytic Continuation & Critical Strip", lecture_lines)
        
        # Setup Complex Plane
        axes = Axes(
            x_range=[-1, 3, 1],
            y_range=[-2, 2, 1],
            axis_config={"include_numbers": True, "font_size": 14}
        )
        self.place_in_area(axes, 'B2', 'E5', scale_factor=0.9)
        
        # Assets
        map_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/map.svg")
        self.place_at_grid(map_icon, 'F6', scale_factor=0.3)
        
        # Critical Strip (0 < Re(s) < 1)
        critical_strip = Polygon(
            axes.c2p(0, -2), axes.c2p(1, -2), axes.c2p(1, 2), axes.c2p(0, 2),
            color="#333333", fill_opacity=0.6, stroke_width=0
        )
        
        # Critical Line (sigma = 1/2)
        critical_line = Line(
            axes.c2p(0.5, -2), axes.c2p(0.5, 2), color=RED, stroke_width=4
        )
        
        zeros = VGroup(*[Dot(axes.c2p(0.5, y), color=YELLOW) for y in [-1.5, -0.5, 0.5, 1.5]])
        for zero in zeros:
            zero.add_updater(lambda z, dt: z.set_opacity(0.5 + 0.5 * np.sin(self.time * 3)))

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.play(Create(axes), FadeIn(map_icon))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(YELLOW))
        self.play(FadeIn(critical_strip))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(RED))
        self.play(Create(critical_line))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[2].animate.set_color(WHITE), self.lecture[3].animate.set_color(GREEN))
        self.play(FadeIn(zeros))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[3].animate.set_color(WHITE), self.lecture[4].animate.set_color(ORANGE))
        self.wait(2)
