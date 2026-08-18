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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines_text = ["Find roots by solving f(x) = 0.", "Use linear approximation with tangent lines.", "Tangent line slope reveals the root.", "The tangent hits the x-axis closer.", "Repeat to hone in on roots."]
        self.setup_layout("Prerequisite: The Tangent Line Intuition", lecture_lines_text)

        # Define Axes
        axes = Axes(
            x_range=[-1, 5, 1], y_range=[-1, 3, 1],
            axis_config={"include_tip": False}
        )
        self.place_in_area(axes, 'B3', 'E6', scale_factor=0.6)
        self.add(axes)

        # Function f(x)
        func = axes.plot(lambda x: 0.2*(x-1)*(x-4) + 1, x_range=[0, 4.5], color=BLUE)
        self.add(func)

        # Asset loading
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        pencil = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pencil.svg")
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")

        # Animation Elements
        x0 = 3.5
        y0 = 0.2*(x0-1)*(x0-4) + 1
        tangent_point = Dot(axes.c2p(x0, y0), color="#FF9900")
        tangent = Line(axes.c2p(2, 1.8), axes.c2p(4.5, -0.5), color="#FFFFFF")
        root_dot = Dot(axes.c2p(4.1, 0), color="#00FF00")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.play(Create(axes), Create(func))

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(BLUE)
        self.place_at_grid(ruler, 'B4', scale_factor=0.3)
        self.play(FadeIn(ruler), Create(tangent))

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FF9900")
        self.place_at_grid(pencil, 'C5', scale_factor=0.3)
        self.place_at_grid(tangent_point, 'C3', scale_factor=0.8)
        self.play(FadeIn(pencil), Create(tangent_point))

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color("#00FF00")
        self.place_at_grid(root_dot, 'D4', scale_factor=0.8)
        self.play(Create(root_dot))

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(BLUE)
        self.place_at_grid(compass, 'E4', scale_factor=0.3)
        self.play(FadeIn(compass))
        self.wait(1)
