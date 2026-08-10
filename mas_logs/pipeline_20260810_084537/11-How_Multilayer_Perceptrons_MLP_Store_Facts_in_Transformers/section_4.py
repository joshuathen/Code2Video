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
            "Training rotates weights toward new data points.",
            "Imagine a vector aligning with a Fact.",
            "Weights minimize the distance to true facts."
        ]
        self.setup_layout("Visualizing Weight Updates", lecture_lines)
        
        # Load Assets
        arrow_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/arrowhead.svg")
        compass_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        
        # Visualization setup
        axes = Axes(x_range=[-2, 2], y_range=[-2, 2], axis_config={"include_tip": True}).scale(0.5)
        self.place_in_area(axes, 'B3', 'D5', scale_factor=0.5)
        
        weight_vec = arrow_icon.copy().scale(0.5).set_color(WHITE)
        fact_vec = compass_icon.copy().scale(0.5).set_color(BLUE)
        
        # Align vectors to origin of axes
        origin = axes.c2p(0, 0)
        weight_vec.move_to(origin + np.array([0.5, 0.25, 0]))
        fact_vec.move_to(origin + np.array([0.25, 0.6, 0]))
        
        vector_group = VGroup(weight_vec, fact_vec)
        self.place_at_grid(vector_group, 'C3', scale_factor=0.7)
        
        self.add(axes, vector_group)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF0000"))
        # Animate rotation toward fact (using lines from center to tip for correct angle)
        target_angle = np.arctan2(*(fact_vec.get_center() - origin)[:2][::-1])
        weight_angle = np.arctan2(*(weight_vec.get_center() - origin)[:2][::-1])
        self.play(Rotate(weight_vec, angle=target_angle - weight_angle, about_point=origin))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        # Pulse effect using final arrow
        final_arrow = arrow_icon.copy().scale(0.5).set_color(GREEN).move_to(weight_vec.get_center())
        self.play(ReplacementTransform(weight_vec, final_arrow))
        self.play(Indicate(final_arrow, color=GREEN))
        self.wait(2)
