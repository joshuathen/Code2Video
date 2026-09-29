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
        self.setup_layout("Mathematical Visualization", [
            "Snell's Law is n1 sin theta1 equals n2 sin theta2.",
            "The beam bends toward or away from normal.",
            "Angles change in real-time as beam rotates."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Graph sin(theta 1) versus sin(theta 2) + prism asset
        prism = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/prism.svg", color=WHITE)
        self.place_at_grid(prism, "A3", scale_factor=0.3)
        self.add(prism)
        
        axes = Axes(x_range=[0, 1, 0.2], y_range=[0, 1, 0.2], axis_config={"include_numbers": False}).scale(0.5)
        self.place_in_area(axes, 'C2', 'E4', scale_factor=0.7)
        self.add(axes)
        
        curve = axes.plot(lambda x: x, color="#FFFFFF")
        self.play(Create(curve), run_time=1)
        self.lecture[0].set_color("#FFFFFF")
        
        # === Animation for Lecture Line 2 ===
        # Highlight slope as n1/n2 + protractor asset
        protractor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/protractor.svg", color="#FF5500")
        self.place_at_grid(protractor, "A5", scale_factor=0.3)
        self.add(protractor)
        
        slope_label = MathTex("n_1/n_2", color="#FF5500").scale(0.7)
        self.place_at_grid(slope_label, 'D5', scale_factor=0.7)
        self.play(Write(slope_label), run_time=1)
        self.lecture[1].set_color("#FF5500")
        
        # === Animation for Lecture Line 3 ===
        # Rotating beam + laser asset
        laser = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg", color="#808080")
        self.place_at_grid(laser, "F2", scale_factor=0.3)
        self.add(laser)
        
        plane = NumberPlane(x_range=[-1, 1], y_range=[-1, 1], axis_config={"include_numbers": False}).scale(0.4)
        self.place_at_grid(plane, 'B4', scale_factor=0.7)
        angle = ValueTracker(PI/4)
        
        # Persistent mobject for the beam, updated via updater
        beam = Line(ORIGIN, 1.0 * RIGHT, color="#808080")
        beam.move_to(plane.get_center())
        beam.add_updater(lambda m: m.set_angle(angle.get_value()))
        self.add(beam)
        
        self.play(Create(plane), run_time=1)
        self.play(angle.animate.set_value(PI/6), run_time=2)
        self.lecture[2].set_color("#808080")
        self.wait(1)
