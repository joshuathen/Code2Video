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
        # Setup title and lecture lines
        lecture_lines = [
            "Swift's position changes over time along this curve.",
            "Let's focus on his movement at two seconds.",
            "Zooming in, the curve eventually looks perfectly straight.",
            "This tangent line reveals the slope at that moment.",
            "It represents Swift's instantaneous speed at that point."
        ]
        self.setup_layout("The Derivative: Capturing the Momentary Change", lecture_lines)

        # Colors
        BLUE_CURVE = "#58C4DD"
        YELLOW_DOT = "#FFFF00"
        RED_TANGENT = "#F91717"
        WHITE_TEXT = "#FFFFFF"

        # Initialize Axes
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 5, 1],
            axis_config={"include_tip": True, "stroke_width": 2},
            x_length=5,
            y_length=5
        )
        # Issue 39 Fix: Avoid overlap by moving axes to B2-F6 with scale 0.6
        self.place_in_area(axes, 'B2', 'F6', scale_factor=0.6)
        
        # Labels for axes
        x_label = Text("t (sec)", font_size=16).next_to(axes.x_axis, DOWN, buff=0.2)
        y_label = Text("s(t) (pos)", font_size=16).next_to(axes.y_axis, LEFT, buff=0.2)
        
        # The Curve: s(t) = 0.25 * t^2 (Rabbit Swift's position)
        curve = axes.plot(lambda t: 0.25 * t**2, x_range=[0, 4.2], color=BLUE_CURVE)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE_CURVE)
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.play(Create(curve), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW_DOT)
        
        dot_pos = axes.c2p(2, 1) # s(2) = 0.25 * 4 = 1
        moment_dot = Dot(point=dot_pos, color=YELLOW_DOT, radius=0.08)
        dot_label = Text("t=2", font_size=20, color=YELLOW_DOT).next_to(moment_dot, UR, buff=0.1)
        
        self.play(FadeIn(moment_dot), Write(dot_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        # Note: No specific color requirement for this step in instructions, keeping default white
        
        # Step 3: Zooming in
        # Group everything to scale around the dot. Labels will move out of the frame.
        zoom_group = VGroup(axes, curve, moment_dot, dot_label, x_label, y_label)
        
        # Perform the zoom effect centered on the dot
        self.play(
            zoom_group.animate.scale(5, about_point=dot_pos),
            run_time=3
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(RED_TANGENT)

        # Step 4: Overlay a red tangent line
        # Tangent for y = 0.25x^2 at x=2 has slope y' = 0.5x = 1.
        # Tangent equation: y - 1 = 1(x - 2) -> y = x - 1
        tangent_line = axes.plot(lambda t: t - 1, x_range=[1.5, 2.5], color=RED_TANGENT)
        
        self.play(Create(tangent_line))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        # Line 5 represents instantaneous speed
        self.lecture[4].set_color(WHITE_TEXT)

        # Issue 40 Fix: speed_text at C5 with scale_factor 0.5 to avoid clutter
        speed_text = Text("s'(2): Instantaneous Speed", font_size=24, color=WHITE_TEXT)
        self.place_at_grid(speed_text, 'C5', scale_factor=0.5)
        
        self.play(Write(speed_text))
        self.wait(2)

        # Reset lecture colors to white for end of section
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(1)
