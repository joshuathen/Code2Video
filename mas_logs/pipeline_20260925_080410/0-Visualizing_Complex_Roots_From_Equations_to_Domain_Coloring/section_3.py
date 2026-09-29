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

class Section3Scene(Scene):
    def construct(self):
        # 1. Setup
        title = Text("Topological Intuition: The Winding Number", font_size=28).to_edge(UP)
        lecture_lines = [
            "A loop's winding number counts its rotations.",
            "If it wraps, a root resides inside.",
            "Like a lasso tightening around a treasure."
        ]
        lecture = VGroup(*[Text(line, font_size=22) for line in lecture_lines])
        lecture.arrange(DOWN, aligned_edge=LEFT).scale(0.8).to_edge(LEFT, buff=0.2)
        
        self.add(title, lecture)

        # 2. Axes and Elements
        axes = Axes(x_length=3, y_length=3, x_range=[-3, 3], y_range=[-3, 3]).shift(RIGHT * 2)
        origin = Dot(axes.c2p(0, 0), color=RED)
        label_o = Text("O", font_size=20, color=RED).next_to(origin, UP + RIGHT, buff=0.1)
        
        treasure = Star(color=YELLOW, fill_opacity=1).scale(0.2).move_to(origin.get_center())
        
        curve = ParametricFunction(
            lambda t: axes.c2p(1.5 * np.cos(t) + 0.5 * np.cos(3 * t), 1.5 * np.sin(t) + 0.5 * np.sin(3 * t)),
            t_range=[0, 2 * PI], color=WHITE
        )
        
        lasso = Circle(radius=0.2, color=BLUE).move_to(curve.get_start())

        # 3. Animations
        self.play(lecture[0].animate.set_color(GREEN))
        self.play(Create(axes), Write(origin), Write(label_o))
        self.play(Create(curve), FadeIn(lasso))
        self.wait(0.5)

        self.play(lecture[1].animate.set_color(GREEN))
        self.play(FadeIn(treasure))
        self.play(MoveAlongPath(lasso, curve), run_time=2, rate_func=linear)
        self.wait(0.5)

        self.play(lecture[2].animate.set_color(GREEN))
        winding_label = Text("W = 1", font_size=24, color=YELLOW).to_edge(RIGHT, buff=1)
        self.play(Write(winding_label))
        self.play(Indicate(winding_label), Indicate(origin))
        self.wait(2)
