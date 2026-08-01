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

class Section3Scene(TeachingScene):
    def construct(self):
        # Setup layout
        lecture_lines = [
            'The complex plane maps numbers in two dimensions.',
            'Real numbers go horizontal; imaginary numbers go vertical.',
            'Every point has a distance and a rotation angle.'
        ]
        self.setup_layout("Prerequisite 2: The Complex Plane", lecture_lines)

        # Colors for highlighting
        color_1 = "#58C4DD" # Sky Blue
        color_2 = "#FFFF00" # Yellow
        color_3 = "#FF8080" # Light Coral
        asset_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/the.svg"

        # === Coordinate System Elements ===
        # Use Arrows for the axes
        x_axis = Arrow(start=self.grid['D1'], end=self.grid['D6'], color=WHITE, buff=0, stroke_width=2)
        y_axis = Arrow(start=self.grid['F3'], end=self.grid['A3'], color=YELLOW, buff=0, stroke_width=2)
        
        # [Asset Integration] Icons for axes tips
        real_axis_icon = SVGMobject(asset_path).set_color(WHITE)
        self.place_at_grid(real_axis_icon, "D6", scale_factor=0.3)
        
        imag_axis_icon = SVGMobject(asset_path).set_color(YELLOW).rotate(90*DEGREES)
        self.place_at_grid(imag_axis_icon, "A3", scale_factor=0.3)

        # Labels (Resolving Issues 38 and 39)
        re_label = Text("Real", color=WHITE, font_size=24)
        self.place_at_grid(re_label, "E6", scale_factor=0.8)
        
        im_label = Text("Imaginary", color=YELLOW, font_size=24)
        self.place_at_grid(im_label, "A4", scale_factor=0.8)

        # === Point and Vector Elements ===
        origin_point = self.grid['D3']
        point_coords = self.grid['C5'] # Corresponds to 2 units right, 1 unit up from D3
        
        point_dot = Dot(point_coords, color=color_3)
        point_label = Text("2 + i", color=color_3, font_size=24)
        self.place_at_grid(point_label, "B5", scale_factor=0.8)
        
        # [Asset Integration] Vector following the asset (Resolving Issue 31)
        # Point is (2,1) relative to origin. Angle is atan(1/2).
        vector_asset = SVGMobject(asset_path).set_color(color_3)
        vector_asset.rotate(np.arctan2(1, 2))
        self.place_in_area(vector_asset, "C3", "D5", scale_factor=0.7)
        
        # Angle and Radius labels
        radius_label = Text("r", color=color_3, font_size=24)
        self.place_at_grid(radius_label, "C4", scale_factor=0.8)
        
        angle_arc = Arc(radius=0.4, start_angle=0, angle=np.arctan2(1, 2), arc_center=origin_point, color=color_3)
        angle_label = Text("θ", color=color_3, font_size=20)
        self.place_at_grid(angle_label, "D4", scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(color_1))
        # Draw Real and Imaginary axes with asset icons
        self.play(Create(x_axis), Create(y_axis), FadeIn(real_axis_icon), FadeIn(imag_axis_icon), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(color_2)
        )
        # Labels for axes
        self.play(Write(re_label), Write(im_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(color_3)
        )
        
        # Moving dot follows the vector path
        moving_dot = Dot(origin_point, color=color_3, radius=0.08)
        self.add(moving_dot)
        
        self.play(
            moving_dot.animate.move_to(point_coords),
            FadeIn(vector_asset),
            run_time=2
        )
        self.play(FadeIn(point_dot), Write(point_label))
        self.play(Create(angle_arc), Write(angle_label), Write(radius_label))
        self.wait(2)

        # Reset colors for final look
        self.play(
            self.lecture[2].animate.set_color(WHITE)
        )
        self.wait(2)
