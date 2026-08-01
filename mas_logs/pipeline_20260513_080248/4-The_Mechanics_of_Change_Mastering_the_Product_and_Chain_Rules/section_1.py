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
        self.setup_layout(
            "Prerequisite Review: The Rate of Change", 
            [
                'Derivatives represent the instantaneous rate of change.', 
                'For f(x) equals x squared, the rate is 2x.', 
                'Complex functions are built from simpler mathematical parts.'
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Derivatives represent the instantaneous rate of change.
        self.play(self.lecture[0].animate.set_color("#FFFFFF"), run_time=0.1)
        
        # Axes to plot parabola
        axes = Axes(
            x_range=[-2, 2, 1],
            y_range=[0, 4, 1],
            x_length=4,
            y_length=4,
            axis_config={"color": "#FFFFFF", "stroke_width": 2}
        )
        # Issue 24 Fix: Adjust axes positioning to avoid vertical crowding
        self.place_in_area(axes, 'C2', 'F5', scale_factor=0.7)
        
        parabola = axes.plot(lambda x: x**2, x_range=[-1.8, 1.8], color="#FFFFFF")
        
        # Point (x, y) at x = 1.0
        x_val = 1.0
        dot_pos = axes.c2p(x_val, x_val**2)
        dot = Dot(dot_pos, color="#FFFFFF")
        label_xy = Text("(x, y)", font_size=18, color="#FFFFFF").next_to(dot, RIGHT, buff=0.1)
        
        self.play(Create(axes), Create(parabola))
        self.play(FadeIn(dot), Write(label_xy))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # For f(x) equals x squared, the rate is 2x.
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        
        # Tangent line at x=1.0, slope = 2*1.0 = 2.
        # Line eq: y = 2(x-1) + 1 => y = 2x - 1
        tangent = axes.plot(lambda x: 2*x - 1, x_range=[0.2, 1.8], color="#FFFF00")
        slope_label = Text("Instantaneous Slope", font_size=18, color="#FFFF00")
        
        # Issue 22 Fix: Change slope_label positioning to prevent cutoff
        self.place_in_area(slope_label, 'A4', 'B5', scale_factor=0.7)
        
        self.play(Create(tangent))
        self.play(Write(slope_label))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Complex functions are built from simpler mathematical parts.
        self.play(self.lecture[2].animate.set_color("#0000FF"))
        
        # Morph graph into a blue square expanding
        square_side = 1.6
        final_square = Square(side_length=square_side, color="#0000FF", fill_opacity=0.3)
        self.place_in_area(final_square, "C3", "E5")
        
        # For the "expanding" effect, we show a smaller square and the growth area
        base_side = 1.4
        dx = 0.2
        base_square = Square(side_length=base_side, color="#0000FF", fill_opacity=0.3)
        self.place_in_area(base_square, "C3", "E5")
        
        # Labels for expansion
        label_x = Text("x", font_size=20, color="#0000FF").next_to(base_square, LEFT, buff=0.1)
        
        # Growth rectangles
        rect_right = Rectangle(width=dx, height=base_side, color="#0000FF", fill_opacity=0.6).next_to(base_square, RIGHT, buff=0)
        rect_top = Rectangle(width=base_side, height=dx, color="#0000FF", fill_opacity=0.6).next_to(base_square, UP, buff=0)
        rect_corner = Square(side_length=dx, color="#0000FF", fill_opacity=0.6).next_to(rect_right, UP, buff=0)
        
        label_dx = Text("dx", font_size=16, color="#0000FF").next_to(rect_right, RIGHT, buff=0.1)
        change_text = Text("Area Change: 2x", font_size=20, color="#0000FF")
        
        # Issue 23 Fix: Reposition change_text to avoid crowding 'dx'
        self.place_in_area(change_text, 'B3', 'B5', scale_factor=0.8)

        # Transition
        self.play(
            FadeOut(axes), FadeOut(parabola), FadeOut(dot), 
            FadeOut(label_xy), FadeOut(tangent), FadeOut(slope_label)
        )
        self.play(Create(base_square), Write(label_x))
        self.wait(0.5)
        self.play(
            FadeIn(rect_right), FadeIn(rect_top), FadeIn(rect_corner),
            Write(label_dx)
        )
        self.play(Write(change_text))
        
        self.wait(3)
