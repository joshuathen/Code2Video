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
            "Complex numbers are vectors in a 2D plane.",
            "Holomorphic dynamics studies point movement under iteration.",
            "Repeated application of a function transforms the plane."
        ]
        self.setup_layout("Prerequisite: Complex Numbers as 2D Spaces", lecture_lines)
        
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/plane.svg] for the complex plane
        # Note: SVG files are loaded as SVGMobject
        try:
            plane_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/plane.svg")
            plane_icon.set_color(WHITE)
            self.place_in_area(plane_icon, "C2", "F6", scale_factor=0.6)
            self.play(FadeIn(plane_icon))
        except:
            # Fallback if asset fails to load
            axes = Axes(
                x_range=[-3, 3, 1],
                y_range=[-3, 3, 1],
                axis_config={"include_tip": True, "color": WHITE}
            ).scale(0.6)
            self.place_in_area(axes, "C2", "F6", scale_factor=0.6)
            self.play(Create(axes))
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        # Using D4 as per requirement
        z_point = Dot(color="#FF00FF")
        self.place_at_grid(z_point, "D4", scale_factor=0.7)
        point_label = Text("z", font_size=24, color="#FF00FF").next_to(z_point, UP)
        
        self.play(FadeIn(z_point), Write(point_label))
        self.play(self.lecture[1].animate.set_color("#FF00FF"))
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        # Vector arrow from origin (implied in background) to D4
        # Need coordinates for origin relative to D4
        vector_arrow = Arrow(start=self.grid["F2"], end=self.grid["D4"], color="#00FFFF", buff=0)
        # Apply the fix-requested scale and positioning
        # self.place_at_grid(vector_arrow, "D4", scale_factor=0.5) 
        
        self.play(GrowArrow(vector_arrow))
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        self.wait(2)
