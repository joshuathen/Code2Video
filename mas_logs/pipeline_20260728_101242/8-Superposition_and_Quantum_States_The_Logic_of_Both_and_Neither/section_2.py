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
        # Title and Lecture Lines
        title = "Prerequisite: The Geometry of States"
        lines = [
            "Quantum states are represented as mathematical vectors.",
            "We define basis states using Dirac Ket notation.",
            "The |0⟩ and |1⟩ states form our coordinate axes."
        ]
        self.setup_layout(title, lines)

        # Colors
        color_0 = "#FFFFFF"  # White
        color_1 = "#FFD700"  # Gold
        color_comb = "#87CEEB"  # Sky Blue

        # Origin for the coordinate system
        origin = self.grid["D3"]

        # === Animation for Lecture Line 1 ===
        # Quantum states are represented as mathematical vectors.
        self.play(self.lecture[0].animate.set_color(YELLOW))

        # Draw a 2D coordinate system with X-axis labeled |1> (#FFD700) and Y-axis labeled |0> (#FFFFFF).
        x_axis = Arrow(start=origin, end=self.grid["D6"], color=color_1, buff=0)
        # Fix Issue 28: self.place_at_grid(label_1, 'D6', scale_factor=0.8)
        label_1 = MathTex(r"|1\rangle", color=color_1, font_size=32)
        self.place_at_grid(label_1, "D6", scale_factor=0.8)

        y_axis = Arrow(start=origin, end=self.grid["A3"], color=color_0, buff=0)
        # Fix Issue 29: self.place_at_grid(label_0, 'A3', scale_factor=0.8)
        label_0 = MathTex(r"|0\rangle", color=color_0, font_size=32)
        self.place_at_grid(label_0, "A3", scale_factor=0.8)

        self.play(Create(x_axis), Create(y_axis))
        self.play(Write(label_1), Write(label_0))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # We define basis states using Dirac Ket notation.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )

        # Animate an arrow (vector) starting from the origin and pointing to (1,0), then to (0,1).
        state_vec = Arrow(start=origin, end=self.grid["D5"], color=WHITE, buff=0, stroke_width=6)
        
        self.play(Create(state_vec))
        self.wait(1)
        self.play(state_vec.animate.put_start_and_end_on(origin, self.grid["B3"]))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The |0⟩ and |1⟩ states form our coordinate axes.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )

        # Animate a new arrow pointing at a 45-degree angle, labeled as a 'combination' of |0> and |1>.
        comb_vec = Arrow(start=origin, end=self.grid["B5"], color=color_comb, buff=0, stroke_width=6)
        
        # Fix Issue 27: self.place_in_area(comb_label, 'A4', 'B6', scale_factor=0.7)
        comb_label = Text("combination", color=color_comb, font_size=24)
        self.place_in_area(comb_label, 'A4', 'B6', scale_factor=0.7)

        self.play(Create(comb_vec), Write(comb_label))
        self.wait(2)

        # Final color reset
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(2)
