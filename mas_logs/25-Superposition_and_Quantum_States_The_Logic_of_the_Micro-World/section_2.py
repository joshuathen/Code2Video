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
        # Initialization
        title_text = "Prerequisite: The Geometry of States"
        lecture_lines = [
            'We map quantum states onto a mathematical two-dimensional plane.',
            'A horizontal vector represents the base state zero.',
            'Rotating the vector creates a new state, Ket psi.'
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Define Colors
        COLOR_AXES = "#FFFFFF"
        COLOR_VECTOR = "#FFFF00"
        COLOR_PSI = "#00FFFF"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_AXES))
        
        # Setup Coordinate Axes
        origin = self.grid["E2"]
        x_target = self.grid["E6"]
        y_target = self.grid["B2"]
        
        x_axis = Arrow(start=origin, end=x_target, color=COLOR_AXES, buff=0)
        y_axis = Arrow(start=origin, end=y_target, color=COLOR_AXES, buff=0)
        
        # Labels for |0> and |1>
        # Issue 37 fix: label_0 at F6
        label_0 = Text("|0⟩", font_size=24, color=COLOR_AXES)
        self.place_at_grid(label_0, "F6", scale_factor=0.8)
        
        # Issue 35 fix: label_1 at A2
        label_1 = Text("|1⟩", font_size=24, color=COLOR_AXES)
        self.place_at_grid(label_1, "A2", scale_factor=0.8)

        self.play(Create(x_axis), Create(y_axis))
        self.play(FadeIn(label_0), FadeIn(label_1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_VECTOR)
        )
        
        # Vector towards |0> position (horizontal)
        # Using E5 as tip to be distinct from axis arrowhead at E6
        vec_0_pos = self.grid["E5"]
        state_vec = Arrow(start=origin, end=vec_0_pos, color=COLOR_VECTOR, buff=0, stroke_width=6)
        
        self.play(GrowArrow(state_vec))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_PSI)
        )
        
        # Rotate/Transform vector to diagonal position (C4)
        diagonal_pos = self.grid["C4"]
        
        # Issue 36 fix: label_psi at B4
        label_psi = Text("|ψ⟩", font_size=24, color=COLOR_PSI)
        self.place_at_grid(label_psi, "B4", scale_factor=0.8)
        
        self.play(
            state_vec.animate.put_start_and_end_on(origin, diagonal_pos),
            FadeIn(label_psi)
        )
        self.wait(3)
