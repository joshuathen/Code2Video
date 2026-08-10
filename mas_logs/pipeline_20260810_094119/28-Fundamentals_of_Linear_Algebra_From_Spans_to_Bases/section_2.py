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
        self.setup_layout("Linear Combinations and Span", [
            "Linear combination sums scaled vectors.",
            "The Span is the reach of all combinations.",
            "Two vectors span a 2D floor.",
            "Robot movements combine to cover the floor.",
            "Span reveals all reachable space points."
        ])

        # Define axes
        axes = Axes(x_range=[-3, 3], y_range=[-3, 3], axis_config={"include_tip": True})
        self.place_in_area(axes, 'B3', 'E6', scale_factor=0.5)
        self.add(axes)

        # Asset: Robot
        robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        self.place_at_grid(robot, 'D3', scale_factor=0.3)

        # Vectors
        u = Arrow(ORIGIN, axes.c2p(2, 1), color="#FF00FF", buff=0)
        v = Arrow(ORIGIN, axes.c2p(-1, 2), color="#00FFFF", buff=0)
        vector_group = VGroup(u, v, robot)
        self.place_in_area(vector_group, 'C3', 'E5', scale_factor=0.45)
        
        v_label = MathTex(r"\\vec{u}", color="#FF00FF")
        w_label = MathTex(r"\\vec{v}", color="#00FFFF")
        self.place_at_grid(w_label, 'B3', scale_factor=0.6)

        # Span area
        span_area = Polygon(
            axes.c2p(0, 0), axes.c2p(2, 1), axes.c2p(1, 3), axes.c2p(-1, 2),
            fill_opacity=0.3, fill_color="#333333", stroke_width=0
        )

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF00FF")
        self.play(Create(u), Write(v_label), FadeIn(robot))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FFFF")
        self.play(Create(v), Write(w_label))

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(span_area))

        # === Animation for Lecture Line 4 ===
        dot = Dot(color=YELLOW)
        dot.move_to(axes.c2p(0, 0))
        self.play(FadeIn(dot))
        self.play(dot.animate.move_to(axes.c2p(0.5, 0.5)), run_time=2)

        # === Animation for Lecture Line 5 ===
        robot_clone = robot.copy()
        self.play(robot_clone.animate.move_to(axes.c2p(1, 2.5)))
        self.play(Indicate(span_area))
        self.wait(1)
