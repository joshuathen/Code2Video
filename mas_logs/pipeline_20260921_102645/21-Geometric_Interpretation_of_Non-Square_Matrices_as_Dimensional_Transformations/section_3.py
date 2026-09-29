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
        lecture_lines = ["A 2x3 matrix maps 3D to 2D.", "This projects 3D space into 2D.", "Information is lost during projection."]
        self.setup_layout("The 'Projection' Scenario: Compressing 3D into 2D", lecture_lines)
        
        # Assets
        grid_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        camera_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg")
        
        # Elements
        grid_3d = grid_icon.copy()
        plane_2d = Rectangle(width=3, height=2, color=BLUE, fill_opacity=0.2)
        target_point_box = Square(side_length=0.5, color=YELLOW)
        point = Dot(color=YELLOW)
        
        # Placement based on fixes
        self.place_at_grid(grid_3d, 'B4', scale_factor=0.5)
        self.place_at_grid(plane_2d, 'D4', scale_factor=0.6)
        self.place_in_area(target_point_box, 'E5', 'F6', scale_factor=0.4)
        point.move_to(grid_3d.get_center())
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#E6E6FA"))
        self.play(FadeIn(grid_3d), Write(point))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#87CEEB"))
        self.play(FadeIn(camera_icon.scale(0.5).next_to(plane_2d, UP)))
        projection_line = Line(point.get_center(), plane_2d.get_center(), color=WHITE, stroke_opacity=0.5)
        self.play(Create(projection_line), point.animate.move_to(plane_2d.get_center()))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF6347"))
        self.play(FadeOut(projection_line), FadeOut(grid_3d), FadeOut(camera_icon))
        self.play(FadeIn(target_point_box))
        self.wait(1)
