from manim import *; config.tex_compiler = "pdflatex"; config.tex_output_format = ".pdf"
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

class Section4Scene(TeachingScene):
    def construct(self):
        # Initial layout setup
        self.setup_layout("Step 2: Isolating the Target (dy/dx)", [
            "We need to isolate the target: dy/dx.",
            "First, subtract 2x from both sides.",
            "Now, divide both sides by 2y.",
            "Twos cancel, leaving -x/y.",
            "This formula calculates slope at any point."
        ])

        # Colors for highlighting lecture lines
        colors = ["#A6E22E", "#F92672", "#66D9EF", "#AE81FF", "#FD971F"]

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(colors[0])
        # Initial equation: 2x + 2y(dy/dx) = 0
        # Fix: Using VGroup of Text to avoid FileNotFoundError: 'latex' in environments without a TeX distribution
        eq1 = VGroup(Text("2x"), Text("+"), Text("2y"), Text("dy/dx"), Text("="), Text("0")).arrange(RIGHT, buff=0.1)
        self.place_in_area(eq1, "B1", "C6", scale_factor=1.5)
        
        self.play(Write(eq1))
        # Highlight the target term dy/dx
        self.play(eq1[3].animate.set_color(colors[0]))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Logic: Subtract 2x from both sides.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(colors[1])
        
        # Next stage: 2y(dy/dx) = -2x
        eq2 = VGroup(Text("2y"), Text("dy/dx"), Text("="), Text("-"), Text("2x")).arrange(RIGHT, buff=0.1)
        self.place_in_area(eq2, "B1", "C6", scale_factor=1.5)
        
        self.play(
            ReplacementTransform(eq1[0], eq2[4]), # Move 2x across the equals sign
            ReplacementTransform(eq1[2], eq2[0]), # Keep 2y
            ReplacementTransform(eq1[3], eq2[1]), # Keep dy/dx
            ReplacementTransform(eq1[4], eq2[2]), # Keep =
            FadeIn(eq2[3], shift=RIGHT),          # Introduce minus sign
            FadeOut(eq1[1]),                      # Remove plus sign
            FadeOut(eq1[5]),                      # Remove 0
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Logic: Divide both sides by 2y.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(colors[2])
        
        # Next stage: dy/dx = -2x / 2y
        eq3 = VGroup(Text("dy/dx"), Text("="), Text("-2x / 2y")).arrange(RIGHT, buff=0.1)
        self.place_in_area(eq3, "B1", "C6", scale_factor=1.5)
        
        self.play(
            ReplacementTransform(eq2[1], eq3[0]), # dy/dx moves to isolate
            ReplacementTransform(eq2[2], eq3[1]), # = stays
            ReplacementTransform(VGroup(eq2[0], eq2[3], eq2[4]), eq3[2]), # 2y, -, and 2x combine
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Logic: Simplify by cancelling the 2s.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(colors[3])
        
        # Result: dy/dx = -x/y
        eq4 = VGroup(Text("dy/dx"), Text("="), Text("-"), Text("x/y")).arrange(RIGHT, buff=0.1)
        self.place_in_area(eq4, "B1", "C6", scale_factor=1.5)
        
        # Visualize simplification
        self.play(
            ReplacementTransform(eq3[0], eq4[0]), 
            ReplacementTransform(eq3[1], eq4[1]), 
            ReplacementTransform(eq3[2], VGroup(*eq4[2:])),
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Logic: The resulting formula represents the slope.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(colors[4])
        
        # Final result boxed in green (#83C167)
        final_box = SurroundingRectangle(eq4, color="#83C167", buff=0.2)
        self.play(Create(final_box))
        self.wait(2)
