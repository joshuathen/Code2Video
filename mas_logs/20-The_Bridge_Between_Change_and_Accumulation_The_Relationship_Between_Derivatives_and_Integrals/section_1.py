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
        title_text = "Prerequisite Review: The Two Faces of a Curve"
        lecture_lines = [
            "Consider a car moving at a constant velocity.",
            "The flat slope means acceleration is exactly zero.",
            "The shaded area represents the total distance traveled."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Define Colors
        BLUE_COLOR = "#58C4DD"
        RED_COLOR = "#F91717"
        GREEN_COLOR = "#87C2A5"

        # Setup Coordinate System
        # Occupies area from B1 to F6
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 4, 1],
            x_length=5,
            y_length=4,
            axis_config={"include_tip": True, "color": WHITE},
            tips=False
        )
        self.place_in_area(axes, "B1", "F6", scale_factor=0.8)
        
        x_label = Text("t", font_size=18).next_to(axes.x_axis, RIGHT, buff=0.1)
        y_label = Text("v(t)", font_size=18).next_to(axes.y_axis, UP, buff=0.1)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE_COLOR))
        
        # Draw a horizontal blue line representing constant velocity
        # v(t) = 2
        velocity_line = axes.plot(lambda t: 2, x_range=[0, 5], color=BLUE_COLOR)
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.play(Create(velocity_line), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(RED_COLOR))
        
        # Add red arrow pointing to the line with text "Slope = 0 (Acceleration)"
        slope_label = Text("Slope = 0\n(Acceleration)", font_size=18, color=RED_COLOR)
        self.place_in_area(slope_label, "C4", "D5", scale_factor=0.8)
        
        slope_point = axes.c2p(2, 2)
        slope_arrow = Arrow(
            start=slope_label.get_left(), 
            end=slope_point, 
            color=RED_COLOR,
            buff=0.1
        )
        
        self.play(GrowArrow(slope_arrow), Write(slope_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN_COLOR))
        
        # Gradually fill area from t=0 to t=4 under the line
        area = axes.get_area(velocity_line, x_range=[0, 4], color=GREEN_COLOR, opacity=0.4)
        area_label = Text("Area (Distance)", font_size=20, color=GREEN_COLOR)
        self.place_in_area(area_label, "E3", "F5", scale_factor=0.7)
        
        self.play(FadeIn(area, shift=UP), run_time=2)
        self.play(Write(area_label))
        self.wait(2)
