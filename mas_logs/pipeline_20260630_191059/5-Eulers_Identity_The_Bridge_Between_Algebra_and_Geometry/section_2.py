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
        # Initialize the layout with title and lecture lines from storyboard
        self.setup_layout(
            "Prerequisite: The Complex Plane and the Secret of 'i'",
            [
                "The complex plane uses real and imaginary number axes.",
                "Multiplying a number by i rotates it ninety degrees.",
                "This operation transforms linear growth into a new direction."
            ]
        )

        # === Animation for Lecture Line 1 ===
        # Highlight first lecture line
        self.play(self.lecture[0].animate.set_color(BLUE))

        # Create the Complex Plane (Axes)
        # Real axis (horizontal, #FFFFFF)
        real_axis = Line(start=LEFT * 2.0, end=RIGHT * 2.0, color="#FFFFFF", stroke_width=2)
        real_label = Text("Real", font_size=18, color="#FFFFFF")
        
        # Imaginary axis (vertical, #FFFF00)
        imag_axis = Line(start=DOWN * 2.0, end=UP * 2.0, color="#FFFF00", stroke_width=2)
        imag_label = Text("Imaginary", font_size=18, color="#FFFF00")
        
        complex_plane_group = VGroup(real_axis, imag_axis)
        
        # Apply the fix from issue #26: B3 to F6, scale 0.7
        self.place_in_area(complex_plane_group, 'B3', 'F6', scale_factor=0.7)
        
        # Reposition labels relative to the now-placed axes
        real_label.next_to(real_axis, RIGHT, buff=0.2)
        imag_label.next_to(imag_axis, UP, buff=0.2)
        
        self.play(
            Create(real_axis), 
            Create(imag_axis), 
            Write(real_label), 
            Write(imag_label),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Transition lecture highlight
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )

        # Place a small green dot (#00FF00) at (1,0)
        origin = complex_plane_group.get_center()
        # real_axis is 4 units wide (from -2 to 2). We want point at 1.0.
        # After scaling by 0.7, distance from origin to 1.0 is 0.7 * 1.0 = 0.7 units.
        unit_len = 1.0 * 0.7
        
        dot = Dot(point=origin + RIGHT * unit_len, color="#00FF00", radius=0.08)
        dot_label = Text("1", font_size=20, color="#00FF00").next_to(dot, DOWN, buff=0.1)
        
        self.play(FadeIn(dot), Write(dot_label))
        self.wait(1)

        # Animate the dot rotating 90 degrees CCW to (0, i) with cyan arc
        arc = Arc(
            radius=unit_len,
            start_angle=0,
            angle=PI/2,
            arc_center=origin,
            color="#00FFFF"
        )
        
        # Target label position (0, i)
        target_pos = origin + UP * unit_len
        i_label = Text("i", font_size=20, slant=ITALIC, color="#00FFFF").next_to(target_pos, LEFT, buff=0.1)

        self.play(
            Rotate(dot, angle=PI/2, about_point=origin),
            Create(arc),
            run_time=2,
            rate_func=smooth
        )
        self.play(Write(i_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Transition lecture highlight
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#00FFFF")
        )
        
        # Pulse the arc to emphasize the change in direction
        self.play(
            arc.animate.set_stroke(width=6),
            rate_func=there_and_back,
            run_time=1.5
        )
        self.wait(2)

        # Final Cleanup for section end
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
