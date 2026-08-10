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
        lecture_lines = [
            "Derivatives linearize complex curves at local points.",
            "Zoom reveals a straight line locally.",
            "Scale factor dictates how inputs transform outputs.",
            "Roller coaster track shows slope's direction.",
            "Small changes in input predict output shifts."
        ]
        self.setup_layout("The Linear Transformation Prism", lecture_lines)
        
        # Setup Axes, Curve and Asset
        axes = Axes(x_range=[-2, 2], y_range=[-2, 8], axis_config={"include_tip": False})
        curve = axes.plot(lambda x: x**2 + 2, color=WHITE)
        rollercoaster = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rollercoaster.svg")
        
        # Combining into a Group
        plot_group = VGroup(axes, curve, rollercoaster)
        # Using place_in_area as requested by issue 19 (grid_group)
        self.place_in_area(plot_group, 'C1', 'F6', scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.play(Create(plot_group))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(GRAY)
        self.play(plot_group.animate.scale(1.5).move_to(self.grid['C3']), run_time=1.5)
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        dx_line = Line(axes.c2p(1, 3), axes.c2p(1.2, 3), color=RED)
        dy_line = Line(axes.c2p(1.2, 3), axes.c2p(1.2, 3.44), color=GREEN)
        triangle = VGroup(dx_line, dy_line)
        self.add(triangle)
        
        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(BLUE)
        # Adding legend as requested in issue 21
        legend_label = Text("Local Linear Approximation", font_size=20, color=YELLOW)
        self.place_at_grid(legend_label, 'B4', scale_factor=0.7)
        self.play(Write(legend_label))
        
        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(GREEN)
        self.play(triangle.animate.shift(RIGHT*0.2 + UP*0.4), run_time=1.5)
        self.wait(1)
