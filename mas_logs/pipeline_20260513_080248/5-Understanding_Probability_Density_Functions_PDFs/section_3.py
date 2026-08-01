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
        # Initial Setup
        title = "Defining the PDF: Height vs. Area"
        lines = [
            "Meet the Probability Density Function, or PDF.",
            "The vertical height f(x) is density, not probability.",
            "Probability is found in the area under the curve.",
            "Shading between 60 and 70 shows that range's probability.",
            "The wider the slice, the more likely the event."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        # Show the smooth curve on axes labeled 'Speed' and 'Density'
        self.lecture[0].set_color(YELLOW)
        
        axes = Axes(
            x_range=[40, 100, 10],
            y_range=[0, 0.1, 0.02],
            axis_config={"include_tip": True, "color": WHITE},
            x_length=5,
            y_length=4
        )
        self.place_in_area(axes, 'B1', 'F6', scale_factor=0.9)
        
        # Simple Gaussian curve for PDF
        curve = axes.plot(
            lambda x: 0.08 * np.exp(-((x - 70) ** 2) / 200),
            color=GREEN
        )
        
        y_label = Text("Density", font_size=20, color=WHITE)
        self.place_at_grid(y_label, 'A2')
        
        x_label = Text("Speed", font_size=20, color=WHITE)
        self.place_at_grid(x_label, 'F6')
        
        self.play(Create(axes), Create(curve), FadeIn(y_label), FadeIn(x_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight height f(x) at a single point.
        self.lecture[1].set_color(YELLOW)
        
        x_point = 85
        y_point = curve.underlying_function(x_point)
        
        v_line = axes.get_vertical_line(axes.c2p(x_point, y_point), color=WHITE, line_func=Line)
        dot = Dot(axes.c2p(x_point, y_point), color=WHITE)
        
        fx_label = Text("f(x)", font_size=18, color=WHITE)
        # Fix for Issue 49: reposition fx_label to A5
        self.place_at_grid(fx_label, 'A5', scale_factor=0.8)
        
        self.play(Create(v_line), FadeIn(dot))
        self.play(Write(fx_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Flash 'Height != Probability' in red (#FF0000)
        self.lecture[2].set_color(YELLOW)
        
        warning_text = Text("Height != Probability", color="#FF0000", font_size=24)
        # Fix for Issue 47: reposition warning_text to A4
        self.place_at_grid(warning_text, 'A4', scale_factor=0.8)
        
        self.play(FadeIn(warning_text))
        self.play(Indicate(warning_text, color="#FF0000", scale_factor=1.2))
        self.wait(1)
        self.play(FadeOut(warning_text), FadeOut(v_line), FadeOut(dot), FadeOut(fx_label))

        # === Animation for Lecture Line 4 ===
        # Mark an interval [a, b] on the X-axis; Shade area between a and b in lime green (#32CD32)
        self.lecture[3].set_color(YELLOW)
        
        a, b = 60, 70
        area = axes.get_area(curve, x_range=[a, b], color="#32CD32", opacity=0.6)
        
        a_label = Text("60", font_size=16, color=WHITE)
        b_label = Text("70", font_size=16, color=WHITE)
        # Manually placing near the x-axis grid points corresponding to the interval
        self.place_at_grid(a_label, 'F3', scale_factor=1.0)
        self.place_at_grid(b_label, 'F4', scale_factor=1.0)
        
        area_prob_label = Text("Area = Probability", font_size=20, color=WHITE)
        # Fix for Issue 48: reposition area_prob_label to area B3-B5
        self.place_in_area(area_prob_label, 'B3', 'B5', scale_factor=0.8)
        
        self.play(FadeIn(area), Write(a_label), Write(b_label))
        self.play(Write(area_prob_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # The wider the slice, the more likely the event.
        self.lecture[4].set_color(YELLOW)
        
        # Expand interval to [55, 85]
        new_a, new_b = 55, 85
        wider_area = axes.get_area(curve, x_range=[new_a, new_b], color="#32CD32", opacity=0.4)
        
        self.play(ReplacementTransform(area, wider_area))
        self.wait(2)
