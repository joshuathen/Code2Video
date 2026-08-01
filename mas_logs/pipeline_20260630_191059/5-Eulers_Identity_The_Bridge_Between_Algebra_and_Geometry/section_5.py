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

class Section5Scene(TeachingScene):
    def construct(self):
        # Titles and lecture lines from storyboard
        title = "The Destination: Plugging in pi"
        lecture_lines = [
            "Plugging in pi creates a half-turn around the circle.",
            "At this point, the value is exactly negative one.",
            "We have reached the opposite side of our start."
        ]
        self.setup_layout(title, lecture_lines)

        # Define colors
        SPRING_GREEN = "#00FF7F"
        BRIGHT_WHITE = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Plugging in pi creates a half-turn around the circle.
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Complex Plane setup
        axes = Axes(
            x_range=[-1.5, 1.5, 1],
            y_range=[-1.5, 1.5, 1],
            x_length=3,
            y_length=3,
            axis_config={"include_tip": True, "color": BLUE_E}
        )
        self.place_in_area(axes, 'B2', 'E5')
        
        circle = Circle(radius=1.0, color=GRAY).move_to(axes.get_center())
        # Re and Im labels - placed strictly in grid cells
        re_label = Text("Re", font_size=16, color=BLUE_E)
        im_label = Text("Im", font_size=16, color=BLUE_E)
        self.place_at_grid(re_label, 'D6', scale_factor=0.8)
        self.place_at_grid(im_label, 'A4', scale_factor=0.8)
        
        # Identity label
        formula_sub = Text("e^(ix) -> e^(i*pi)", font_size=24, color=YELLOW)
        self.place_at_grid(formula_sub, 'A3')

        self.play(Create(axes), Write(re_label), Write(im_label), Create(circle))
        self.play(Write(formula_sub))
        
        theta_tracker = ValueTracker(0)
        
        # The moving point
        dot = Dot(color=WHITE)
        dot.add_updater(lambda d: d.move_to(axes.c2p(np.cos(theta_tracker.get_value()), np.sin(theta_tracker.get_value()))))
        
        # The path (arc)
        arc = always_redraw(lambda: Arc(
            radius=1.0, 
            start_angle=0, 
            angle=theta_tracker.get_value(), 
            color=YELLOW
        ).move_to(axes.get_center()))
        
        self.add(dot, arc)
        
        # Rotate halfway (pi radians)
        self.play(theta_tracker.animate.set_value(PI), run_time=3, rate_func=smooth)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # At this point, the value is exactly negative one.
        self.play(self.lecture[1].animate.set_color(SPRING_GREEN))
        
        # Pulse at destination
        pulse = Circle(radius=0.1, color=BRIGHT_WHITE, stroke_width=4).move_to(dot.get_center())
        self.play(pulse.animate.scale(4).set_opacity(0), run_time=1)
        self.remove(pulse)

        # cos(pi) = -1 and sin(pi) = 0
        cos_text = Text("cos(pi) = -1", color=SPRING_GREEN, font_size=22)
        sin_text = Text("sin(pi) = 0", color=SPRING_GREEN, font_size=22)
        
        # Apply Issue Fixes for positioning
        self.place_at_grid(cos_text, 'B6', scale_factor=0.8)
        self.place_at_grid(sin_text, 'C6', scale_factor=0.8)
        
        self.play(Write(cos_text), Write(sin_text))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # We have reached the opposite side of our start.
        self.play(self.lecture[2].animate.set_color(BRIGHT_WHITE))
        
        final_eq = Text("e^(i*pi) = -1", color=BRIGHT_WHITE, font_size=32)
        # Apply Issue Fix for positioning
        self.place_in_area(final_eq, 'F3', 'F5', scale_factor=1.0)
        
        # Box for finalized equation
        box = SurroundingRectangle(final_eq, color=BRIGHT_WHITE, buff=0.1)
        
        self.play(
            FadeOut(formula_sub),
            FadeOut(cos_text),
            FadeOut(sin_text),
            Write(final_eq),
            Create(box)
        )
        self.wait(2)
