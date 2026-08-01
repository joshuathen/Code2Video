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
        title_text = "The Foundation: Prerequisite - What is a Slope?"
        lines = [
            "Imagine a blue plank on a coordinate grid.",
            "Turbo moves along the line from start to finish.",
            "A triangle shows he rises two for every one.",
            "This constant rate of change is the slope, two.",
            "But what happens if the path is a curve?"
        ]
        self.setup_layout(title_text, lines)
        
        # === Animation for Lecture Line 1 ===
        # Highlight current line with matching blue color
        self.lecture[0].set_color("#0000FF")
        
        # Display coordinate grid with blue line y=2x
        axes = Axes(
            x_range=[0, 3, 1],
            y_range=[0, 5, 1],
            axis_config={"include_tip": True, "color": GREY},
            x_length=4,
            y_length=4
        )
        self.place_in_area(axes, "B1", "F6")
        
        # y = 2x line in Blue (#0000FF)
        plank = axes.plot(lambda x: 2 * x, x_range=[0, 2], color="#0000FF")
        
        self.play(Create(axes), Create(plank), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Update highlighting
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(WHITE) # Matching Turbo color
        
        # Move a small snail icon 'Turbo' along the line segment from (0,0) to (1,2)
        turbo = Dot(axes.c2p(0, 0), color=WHITE, radius=0.1)
        turbo_label = Text("Turbo", font_size=16).next_to(turbo, UP, buff=0.1)
        
        def update_label(m):
            m.next_to(turbo, UP, buff=0.1)
        turbo_label.add_updater(update_label)
        
        self.play(FadeIn(turbo), FadeIn(turbo_label))
        self.play(turbo.animate.move_to(axes.c2p(1, 2)), run_time=2)
        turbo_label.remove_updater(update_label)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Update highlighting
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FF0000") # Matching red triangle color
        
        # Draw a red (#FF0000) right triangle highlighting 'Rise=2' and 'Run=1'
        p0 = axes.c2p(0, 0)
        p1 = axes.c2p(1, 0)
        p2 = axes.c2p(1, 2)
        
        run_line = Line(p0, p1, color="#FF0000")
        rise_line = Line(p1, p2, color="#FF0000")
        run_label = Text("Run=1", font_size=14, color="#FF0000").next_to(run_line, DOWN, buff=0.1)
        rise_label = Text("Rise=2", font_size=14, color="#FF0000").next_to(rise_line, RIGHT, buff=0.1)
        triangle = VGroup(run_line, rise_line, run_label, rise_label)
        
        self.play(Create(triangle), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Update highlighting
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color("#FFFF00") # Matching yellow slope text color
        
        # Show the text 'Slope = 2' in yellow (#FFFF00) next to the triangle
        # Fix from Issue 30: position at A5, scale 1.2
        slope_text = Text("Slope = 2", color="#FFFF00", font_size=24)
        self.place_at_grid(slope_text, "A5", scale_factor=1.2)
        
        self.play(Write(slope_text))
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        # Update highlighting
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color("#00FF00") # Matching green dot color
        
        # Morph the line into a white parabola
        parabola = axes.plot(lambda x: x**2, x_range=[0, 2.2], color=WHITE)
        
        # Mark a point with a blinking green (#00FF00) dot
        blink_dot = Dot(axes.c2p(1.5, 2.25), color="#00FF00")
        
        self.play(
            FadeOut(triangle),
            FadeOut(turbo),
            FadeOut(turbo_label),
            FadeOut(slope_text),
            Transform(plank, parabola),
            run_time=2
        )
        
        # Blinking effect for the green dot
        self.add(blink_dot)
        for _ in range(3):
            self.play(blink_dot.animate.set_opacity(0), run_time=0.3)
            self.play(blink_dot.animate.set_opacity(1), run_time=0.3)
            
        self.wait(1)
