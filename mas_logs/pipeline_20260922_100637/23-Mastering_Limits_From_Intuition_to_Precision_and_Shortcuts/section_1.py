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
        self.setup_layout("The Intuitive Foundation: Closing the Gap", [
            "Limits track behavior, not the point value.",
            "A robot vacuum approaches, but never touches.",
            "As x nears c, y nears L.",
            "The hole remains, but the target is clear.",
            "Convergence happens regardless of the destination."
        ])
        
        # Setup Axes and Graph
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 4, 1], axis_config={"include_tip": True}).scale(0.5)
        self.place_in_area(axes, "C2", "E5", scale_factor=0.6)
        
        graph = axes.plot(lambda x: x if x != 2 else 5, x_range=[0, 4])
        hole = Circle(radius=0.1, color=WHITE).move_to(axes.c2p(2, 2))
        
        vacuum = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/vacuum.svg")
        self.place_at_grid(vacuum, "F1", scale_factor=0.3)
        
        # === Animation for Lecture Line 1 ===
        self.add(graph, hole)
        self.play(FadeIn(vacuum))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF00FF")
        # Animate vacuum moving towards x=2 (hole)
        path = Line(start=vacuum.get_center(), end=axes.c2p(2, 0))
        self.play(MoveAlongPath(vacuum, path), run_time=2)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FFFF")
        y_label = DashedLine(axes.c2p(2, 0), axes.c2p(2, 2), color="#00FFFF")
        self.play(Create(y_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFFF00")
        self.play(Indicate(hole))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#00FF00")
        limit_text = Text("Limit notation: L", color="#00FF00").scale(0.6)
        self.place_at_grid(limit_text, "A3")
        self.play(Write(limit_text))
        self.wait(2)
