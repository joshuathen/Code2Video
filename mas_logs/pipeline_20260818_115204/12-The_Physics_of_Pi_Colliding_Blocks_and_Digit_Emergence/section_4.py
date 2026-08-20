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
            "Trajectory unfolds as mass ratio grows.",
            "Zigzag path smooths into circle arc.",
            "Path length reveals ratio of Pi."
        ]
        self.setup_layout("The Limit Case: Why Pi?", lecture_lines)
        
        # Assets
        circle_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg")
        self.place_at_grid(circle_icon, "A5", scale_factor=0.5)
        
        # Objects
        # 1. Coordinate system/Phase space
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 4, 1], axis_config={"include_tip": False})
        self.place_in_area(axes, "B2", "E5", scale_factor=0.55)

        # 2. Path (initially a rough path)
        path = VMobject()
        path.set_points_smoothly([axes.c2p(0,0), axes.c2p(1,1), axes.c2p(1,2), axes.c2p(2,2), axes.c2p(2,3), axes.c2p(3,3)])
        path.set_stroke(BLUE, width=4)

        # 3. Pi text
        pi_text = MathTex(r"Path \approx \pi", color=GREEN)
        self.place_at_grid(pi_text, "E5", scale_factor=1.2)
        pi_text.set_opacity(0)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.play(Create(axes), Create(path), FadeIn(circle_icon))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color("#FFFFFF"))
        # Transform path to arc
        arc = Arc(radius=axes.c2p(2,0)[0]-axes.c2p(0,0)[0], start_angle=0, angle=PI/2, arc_center=axes.c2p(0,0), stroke_color=RED)
        self.play(Transform(path, arc))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color("#00FF00"))
        self.play(FadeIn(pi_text))
        self.play(pi_text.animate.set_opacity(1))
        self.wait(2)
