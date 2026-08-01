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
        title_str = "Prerequisite: Antipodal Points and Mapping"
        lecture_lines = [
            'Antipodal points are directly opposite on a circle or sphere.',
            'Every point x has an opposite partner -x.',
            'We can map these points to values like temperature.'
        ]
        self.setup_layout(title_str, lecture_lines)

        # Colors for highlights
        color_1 = "#64B5F6"  # Blue
        color_2 = "#81C784"  # Green
        color_3 = "#FF8A65"  # Coral

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(color_1))
        
        circle = Circle(radius=1.8, color=WHITE)
        self.place_in_area(circle, "B2", "E5")
        
        # Center point for reference (invisible or small)
        center_pt = circle.get_center()
        
        # Point x at an angle
        angle_x = 30 * DEGREES
        pos_x = center_pt + circle.radius * np.array([np.cos(angle_x), np.sin(angle_x), 0])
        dot_x = Dot(pos_x, color=color_1)
        label_x = Text("x", color=color_1, font_size=30).next_to(dot_x, UR, buff=0.1)
        
        self.play(Create(circle))
        self.play(FadeIn(dot_x), Write(label_x))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(color_2)
        )
        
        # Antipodal point -x
        angle_nx = angle_x + PI
        pos_nx = center_pt + circle.radius * np.array([np.cos(angle_nx), np.sin(angle_nx), 0])
        dot_nx = Dot(pos_nx, color=color_2)
        label_nx = Text("-x", color=color_2, font_size=30).next_to(dot_nx, DL, buff=0.1)
        
        # Dashed line segment through center
        diameter_line = DashedLine(pos_x, pos_nx, color=GRAY, dash_length=0.1)
        
        self.play(Create(diameter_line))
        self.play(FadeIn(dot_nx), Write(label_nx))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(color_3)
        )
        
        # Temperature gradient around the circle
        # We create a colored arc or simply color the circle stroke
        temp_circle = Circle(radius=1.8, stroke_width=8).set_color_gradient([BLUE, RED, BLUE])
        self.place_in_area(temp_circle, "B2", "E5")
        
        temp_label = Text("f(x) = Temperature", font_size=20, color=color_3)
        # Fix for issue 30 and 31: Use place_in_area for multi-word labels and align horizontally
        self.place_in_area(temp_label, 'F2', 'F5', scale_factor=0.8)
        
        self.play(
            circle.animate.set_stroke(opacity=0.3),
            Create(temp_circle),
            FadeIn(temp_label)
        )
        
        # Pulse the temperature circle to show the "mapping"
        self.play(temp_circle.animate.set_stroke(width=12), run_time=1)
        self.play(temp_circle.animate.set_stroke(width=8), run_time=1)
        
        self.wait(3)
