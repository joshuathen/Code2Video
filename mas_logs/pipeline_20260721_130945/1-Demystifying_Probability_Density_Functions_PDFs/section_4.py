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
        # === Setup ===
        title_text = "Character Application: Felix the Archer Fox"
        lecture_lines = [
            "- Felix the Fox shoots arrows at a target.",
            "- A bell curve shows his accuracy near center.",
            "- Shading the curve predicts his chance of success."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        felix_color = "#FFFFFF"
        target_color = "#FF0000"
        curve_color = "#00FFFF"
        shade_color = "#00FF00"
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(felix_color)
        
        # Felix (Simple procedural fox shape)
        ear_l = Triangle().scale(0.12).rotate(15*DEGREES).set_fill(felix_color, 1).set_stroke(WHITE, 1)
        ear_r = Triangle().scale(0.12).rotate(-15*DEGREES).set_fill(felix_color, 1).set_stroke(WHITE, 1)
        ear_l.shift(LEFT*0.12 + UP*0.18)
        ear_r.shift(RIGHT*0.12 + UP*0.18)
        face = Circle(radius=0.22).set_fill(felix_color, 1).set_stroke(WHITE, 1)
        felix = VGroup(ear_l, ear_r, face)
        
        # [Fix Issue 29]: Position Felix at B4 to be visually connected and centered above target
        self.place_at_grid(felix, "B4", scale_factor=0.8)
        
        # Target (Concentric circles)
        target = VGroup(
            Circle(radius=1.5, color=target_color).set_fill(target_color, 0.2),
            Circle(radius=1.0, color=WHITE).set_fill(WHITE, 0.2),
            Circle(radius=0.5, color=target_color).set_fill(target_color, 0.4),
            Circle(radius=0.1, color=WHITE).set_fill(WHITE, 1)
        )
        
        # Axes
        axes = Axes(
            x_range=[-10, 10, 5],
            y_range=[0, 1, 0.5],
            axis_config={"include_tip": False, "stroke_width": 2},
            x_length=5,
            y_length=3
        )
        
        vis_group = VGroup(target, axes)
        # [Fix Issue 28]: Position vis_group in area C3 to E5 to avoid lecture note overlap
        self.place_in_area(vis_group, "C3", "E5", scale_factor=0.7)
        
        self.play(FadeIn(felix), FadeIn(target))
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(curve_color)
        
        # Bell curve - centered on bullseye
        # Using a Gaussian function to represent PDF
        curve = axes.plot(lambda x: 0.8 * np.exp(-(x/3.5)**2), color=curve_color, x_range=[-10, 10])
        
        self.play(Create(axes), Create(curve))
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(shade_color)
        
        # Shade Success area: bullseye center is 0, Success is +/- 5
        shade = axes.get_area(curve, x_range=[-5, 5], color=shade_color, opacity=0.4)
        
        # Label "Success"
        success_label = Text("Success", font_size=24, color=shade_color)
        # [Fix Issue 27]: Position Success label at F5 and scale to 0.6 to avoid overlap and clipping
        self.place_at_grid(success_label, "F5", scale_factor=0.6)
        
        self.play(FadeIn(shade), Write(success_label))
        self.play(Indicate(shade))
        self.wait(2)
