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
        title_text = "The Grand Reveal: Snell’s Law"
        lines = [
            "sin(theta1) / v1 = sin(theta2) / v2.",
            "Multiply by c to introduce refractive indices n1 and n2.",
            "Snell’s Law: n1 sin(theta1) = n2 sin(theta2)."
        ]
        self.setup_layout(title_text, lines)
        
        # === Animation for Lecture Line 1 ===
        # Color the first line to indicate focus
        self.play(self.lecture[0].animate.set_color(BLUE))
        
        # Display sin(theta 1) / v1 = sin(theta 2) / v2
        eq1 = VGroup(
            Text("sin(theta1) / "), # Index 0
            Text("v1"),              # Index 1
            Text(" = sin(theta2) / "), # Index 2
            Text("v2")               # Index 3
        ).arrange(RIGHT, buff=0.1)
        
        # Position in a safe area (B2-B5) to avoid title crowding
        self.place_in_area(eq1, 'B2', 'B5', scale_factor=0.9)
        
        self.play(Write(eq1))
        self.wait(1)
        
        # Highlight v1 and v2 in yellow
        self.play(
            eq1[1].animate.set_color(YELLOW),
            eq1[3].animate.set_color(YELLOW)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Color the second line to indicate focus
        self.play(self.lecture[1].animate.set_color(GREEN))
        
        # Show 'c' appearing in the numerators
        eq2 = VGroup(
            Text("c sin(theta1) / "),
            Text("v1"),
            Text(" = c sin(theta2) / "),
            Text("v2")
        ).arrange(RIGHT, buff=0.1)
        self.place_in_area(eq2, 'B2', 'B5', scale_factor=0.9)
        
        # Introduce refractive index definitions
        n1_def = Text("n1 = c / v1").set_color(GREEN)
        n2_def = Text("n2 = c / v2").set_color(GREEN)
        defs = VGroup(n1_def, n2_def).arrange(RIGHT, buff=0.8)
        self.place_at_grid(defs, 'D3', scale_factor=0.8)
        
        # Animate the transition to 'c' and the appearance of definitions
        self.play(Transform(eq1, eq2), run_time=1.5)
        self.play(FadeIn(defs, shift=UP))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Color the third line to indicate focus in Gold
        self.play(self.lecture[2].animate.set_color(GOLD))
        
        # Final Snell's Law equation
        eq3 = Text("n1 sin(theta1) = n2 sin(theta2)", font_size=28, color=GOLD)
        self.place_in_area(eq3, 'B2', 'B5', scale_factor=0.9)
        
        # Highlight the reveal
        rect = SurroundingRectangle(eq3, color=GOLD, buff=0.3)
        
        self.play(
            ReplacementTransform(eq1, eq3),
            FadeOut(defs),
            run_time=2
        )
        self.play(Create(rect))
        self.wait(3)
