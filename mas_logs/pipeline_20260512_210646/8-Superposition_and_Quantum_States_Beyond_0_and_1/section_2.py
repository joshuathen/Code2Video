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
        # Setup the title and lecture lines
        lecture_lines = [
            'We denote quantum states using "Ket" notation.',
            'States live in a two-dimensional vector space.',
            'A vertical vector represents the zero state.',
            'A horizontal vector represents the one state.',
            'Any diagonal vector shows a combination of both.'
        ]
        self.setup_layout("The Mathematical Canvas: State Vectors", lecture_lines)

        # Define grid references for cleaner logic
        origin_pos = self.grid['E2']
        y_axis_end = self.grid['B2']
        x_axis_end = self.grid['E5']
        diag_end = self.grid['C4']

        # === Animation for Lecture Line 1 ===
        # Stage: Ket_Symbol. Animation: The symbol |ψ⟩ fades into the top-center of the screen in light green (#ADFF2F).
        ket_symbol = Text("|ψ⟩", color="#ADFF2F", font_size=48)
        # Resolved Issue 30: Moved from A3-B4 to A5-B6
        self.place_in_area(ket_symbol, 'A5', 'B6', scale_factor=0.8)
        
        self.play(
            self.lecture[0].animate.set_color("#ADFF2F"),
            FadeIn(ket_symbol)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Stage: Vector_Space. Animation: Two perpendicular white arrows (#FFFFFF) appear from a central origin.
        y_axis = Arrow(start=origin_pos, end=y_axis_end, buff=0, color="#FFFFFF")
        x_axis = Arrow(start=origin_pos, end=x_axis_end, buff=0, color="#FFFFFF")
        
        self.play(
            self.lecture[1].animate.set_color("#FFFFFF"),
            GrowArrow(y_axis),
            GrowArrow(x_axis)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Stage: Basis_0. Animation: Label the vertical arrow |0⟩ in orange-red (#FF4500).
        label_0 = Text("|0⟩", color="#FF4500", font_size=32)
        self.place_at_grid(label_0, 'A2') # Positioned above the vertical arrow end
        
        self.play(
            self.lecture[2].animate.set_color("#FF4500"),
            Write(label_0),
            y_axis.animate.set_color("#FF4500")
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Stage: Basis_1. Animation: Label the horizontal arrow |1⟩ in dodger blue (#1E90FF).
        label_1 = Text("|1⟩", color="#1E90FF", font_size=32)
        # Resolved Issue 32: scale_factor adjusted to 0.8
        self.place_at_grid(label_1, 'E6', scale_factor=0.8) # Positioned to the right of the horizontal arrow end
        
        self.play(
            self.lecture[3].animate.set_color("#1E90FF"),
            Write(label_1),
            x_axis.animate.set_color("#1E90FF")
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Stage: State_Vector. Animation: A new yellow vector (#FFFF00) appears, pointing diagonally.
        state_vector = Arrow(start=origin_pos, end=diag_end, buff=0, color="#FFFF00")
        state_label = Text("|ψ⟩", color="#FFFF00", font_size=32)
        # Resolved Issue 31: scale_factor adjusted to 0.7
        self.place_at_grid(state_label, 'B4', scale_factor=0.7) # Near the end of the diagonal vector
        
        self.play(
            self.lecture[4].animate.set_color("#FFFF00"),
            GrowArrow(state_vector),
            Write(state_label)
        )
        self.wait(2)
