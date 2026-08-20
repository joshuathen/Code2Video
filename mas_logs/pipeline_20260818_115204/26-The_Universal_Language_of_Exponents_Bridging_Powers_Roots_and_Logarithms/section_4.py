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
        self.setup_layout("Visual Synthesis: The Transformation Map", [
            "Use our triangular relationship map.",
            "Powers, roots, logs are linked.",
            "Rotate views to see connections.",
            "2^3=8 connects these three forms.",
            "Perspectives change, the math stays."
        ])
        
        axes = Axes(x_range=[0.1, 4, 1], y_range=[0.1, 4, 1], axis_config={"include_tip": True})
        # Applying fix for Issue 29
        self.place_in_area(axes, 'D2', 'F6', scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FFCC")
        exp_graph = axes.plot(lambda x: 2**x if 2**x < 4 else 4, color="#00FFCC")
        self.play(Create(exp_graph))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF99FF")
        log_graph = axes.plot(lambda x: np.log2(x), color="#FF99FF")
        self.play(Create(log_graph))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(WHITE)
        line_xy = axes.plot(lambda x: x, color=WHITE, stroke_width=2, stroke_opacity=0.5)
        self.play(Create(line_xy))
        
        # Preparing curve group for Issue 31
        curve_group = VGroup(exp_graph, log_graph, line_xy)
        self.place_in_area(curve_group, 'C2', 'F5', scale_factor=0.75)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFCC00")
        point = Dot(axes.c2p(2, 2), color="#FFCC00")
        # Asset integration (none.svg placeholder)
        asset_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        formula_label = Text("2^1=2", font_size=18, color="#FFCC00")
        
        # Applying fix for Issue 30
        self.place_at_grid(formula_label, 'D3', scale_factor=0.7)
        self.play(Create(point), Write(formula_label), FadeIn(asset_icon.next_to(point, RIGHT, buff=0.1)))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(GREEN)
        self.wait(1)
