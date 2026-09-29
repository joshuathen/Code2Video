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
        self.setup_layout("Prerequisite: The Geometric Interpretation", [
            "Derivatives represent the tangent line slope.",
            "Imagine a climber on a steep curve.",
            "The slope reflects local steepness changes."
        ])
        
        # Setup curve and objects
        axes = Axes(x_range=[0, 4], y_range=[0, 4], axis_config={"include_tip": False})
        curve = FunctionGraph(lambda x: 0.2 * x**3, x_range=[0, 3.5], color=BLUE)
        point = Dot(curve.point_from_proportion(0.6), color=WHITE)
        
        # For animations
        tangent_line = Line(start=ORIGIN, end=RIGHT*2, color="#FF0000")
        
        self.place_in_area(axes, "A3", "F6", scale_factor=0.5)
        self.add(axes)
        self.add(curve)
        self.add(point)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF0000")
        # Draw tangent line at point on curve
        tangent_line.move_to(point.get_center())
        self.play(Create(tangent_line))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FF00")
        # Highlight slope calculation triangle
        triangle = Polygon(
            point.get_center(), 
            point.get_center() + RIGHT*0.5, 
            point.get_center() + RIGHT*0.5 + UP*0.5, 
            color="#00FF00"
        )
        self.play(Create(triangle))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFFFF")
        # Animate secant line approaching tangent line
        secant = Line(start=curve.point_from_proportion(0.4), end=curve.point_from_proportion(0.8), color=GRAY)
        self.play(Create(secant))
        self.play(secant.animate.move_to(tangent_line.get_center()), run_time=2)
        
        # Add labels
        slope_label = Text("Slope = Rise/Run", font_size=20, color="#FFFF00")
        self.place_at_grid(slope_label, "B3", scale_factor=0.7)
        self.play(Write(slope_label))
        
        slope_val = Text("m = 2.5", font_size=20, color="#00FFFF")
        self.place_at_grid(slope_val, "C3", scale_factor=0.7)
        self.play(Indicate(slope_val))
        self.wait(2)
