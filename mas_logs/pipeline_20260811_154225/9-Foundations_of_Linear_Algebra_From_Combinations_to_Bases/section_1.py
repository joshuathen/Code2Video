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
        lecture_lines = [
            "Scale two vectors to build combinations.",
            "Combine them to sweep the plane.",
            "All reachable points form the span.",
            "Think: A robot arm's workspace.",
            "Span is the set of all reaches."
        ]
        self.setup_layout("The Ingredients: Linear Combinations and Span", lecture_lines)
        
        axes = Axes(x_range=[-3, 3], y_range=[-3, 3], axis_config={"include_tip": True}).scale(0.4)
        self.place_in_area(axes, 'C2', 'E5', scale_factor=0.6)

        v1 = Vector([1, 2], color="#FF5733")
        v2 = Vector([2, -1], color="#33FF57")
        l1 = MathTex(r"v_1", color="#FF5733")
        l2 = MathTex(r"v_2", color="#33FF57")
        
        self.place_at_grid(l1, 'C1', scale_factor=0.7)
        self.place_at_grid(l2, 'F3', scale_factor=0.7)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF5733"), FadeIn(v1), FadeIn(l1), FadeIn(v2), FadeIn(l2))

        # === Animation for Lecture Line 2 ===
        v3 = Vector([3, 1], color="#3357FF")
        sum_label = MathTex(r"Sum", color="#3357FF")
        self.place_at_grid(sum_label, 'D6', scale_factor=0.7)
        
        # Area for span
        span_area = Polygon(ORIGIN, [1, 2, 0], [3, 1, 0], [2, -1, 0], color=WHITE, fill_opacity=0.3)
        self.place_in_area(span_area, 'C2', 'E5', scale_factor=0.4)
        
        self.play(self.lecture[1].animate.set_color("#33FF57"), FadeIn(v3), FadeIn(sum_label), FadeIn(span_area))

        # === Animation for Lecture Line 3 ===
        span_label = Text("Span", color=WHITE).scale(0.7)
        self.place_at_grid(span_label, 'B5', scale_factor=0.7)
        self.play(self.lecture[2].animate.set_color("#3357FF"), Write(span_label))

        # === Animation for Lecture Line 4 ===
        robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        self.place_at_grid(robot, 'E2', scale_factor=0.5)
        self.play(self.lecture[3].animate.set_color(YELLOW), FadeIn(robot))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(1)
