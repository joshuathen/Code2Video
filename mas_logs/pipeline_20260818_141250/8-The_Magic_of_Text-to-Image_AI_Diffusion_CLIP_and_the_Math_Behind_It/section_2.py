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

class Section2Scene(ThreeDScene, TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisites: High-Dimensional Spaces", [
            "Images exist in high-dimensional space.",
            "Each pixel contributes to a vector.",
            "Embeddings map features to coordinates."
        ])
        
        # === Animation for Lecture Line 1 ===
        axes = Axes(x_range=[0, 5, 1], y_range=[0, 5, 1], axis_config={"color": "#808080"}).scale(0.5)
        self.place_at_grid(axes, 'D3', scale_factor=0.6)
        self.play(Create(axes))
        self.lecture[0].set_color("#808080")

        # === Animation for Lecture Line 2 ===
        sensor_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sensor.svg", color=WHITE)
        vector = Arrow(start=ORIGIN, end=axes.c2p(3, 4), buff=0, color="#FF4500")
        dot = Dot(axes.c2p(3, 4), color="#FF4500")
        self.place_at_grid(sensor_icon, 'B3', scale_factor=0.5)
        
        self.play(Create(vector), Create(dot), FadeIn(sensor_icon))
        self.lecture[1].set_color("#FF4500")

        # Expand to 3D space with subtle rotation
        axes_3d = ThreeDAxes(x_range=[0, 5, 1], y_range=[0, 5, 1], z_range=[0, 5, 1], axis_config={"color": "#808080"}).scale(0.4)
        camera_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg", color=WHITE)
        self.place_at_grid(axes_3d, 'D4', scale_factor=0.6)
        self.place_at_grid(camera_icon, 'B5', scale_factor=0.5)

        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)
        self.play(FadeOut(axes), FadeOut(vector), FadeOut(dot), FadeOut(sensor_icon), Create(axes_3d), FadeIn(camera_icon))
        
        # === Animation for Lecture Line 3 ===
        vec_3d = Arrow3D(start=ORIGIN, end=axes_3d.c2p(3, 4, 3), color="#32CD32")
        self.play(Create(vec_3d))
        self.lecture[2].set_color("#32CD32")
        
        dot_a = Dot(axes_3d.c2p(3, 4, 3), color="#FF69B4")
        dot_b = Dot(axes_3d.c2p(1, 1, 1), color="#FF69B4")
        line_dist = Line3D(start=dot_a.get_center(), end=dot_b.get_center(), color="#FF69B4")
        
        graph_group = VGroup(vec_3d, dot_a, dot_b, line_dist)
        self.play(Create(dot_a), Create(dot_b), Create(line_dist))
        self.wait(2)
