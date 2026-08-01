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
        # Data from shared state
        title_text = "The Leap: Defining e^A"
        lecture_lines = [
            "We can extend this series to square matrices.",
            "Replace every scalar variable x with matrix A.",
            "The number one becomes the identity matrix I.",
            "Higher powers of A follow standard matrix multiplication.",
            "The resulting sum defines the matrix exponential."
        ]
        
        self.setup_layout(title_text, lecture_lines)

        # Colors
        COLOR_IDENTITY = "#00FFFF" # Cyan
        COLOR_EQ_BLUE = BLUE
        COLOR_EQ_YELLOW = YELLOW

        # === Animation for Lecture Line 1 ===
        # "We can extend this series to square matrices."
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        scalar_series = Text("e^x = 1 + x + x^2/2! + x^3/3! + ...", font_size=32)
        self.place_at_grid(scalar_series, "B3", scale_factor=1.0)
        self.play(Write(scalar_series))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Replace every scalar variable x with matrix A."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        matrix_series_v1 = Text("e^A = 1 + A + A^2/2! + A^3/3! + ...", font_size=32)
        self.place_at_grid(matrix_series_v1, "C3", scale_factor=1.0)
        
        self.play(TransformFromCopy(scalar_series, matrix_series_v1))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "The number one becomes the identity matrix I."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_IDENTITY)
        )
        
        matrix_series_v2 = Text("e^A = I + A + A^2/2! + A^3/3! + ...", font_size=32,
                                t2c={"I": COLOR_IDENTITY})
        self.place_at_grid(matrix_series_v2, "C3", scale_factor=1.0)
        
        self.play(Transform(matrix_series_v1, matrix_series_v2))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "Higher powers of A follow standard matrix multiplication."
        self.play(
            self.lecture[3].animate.set_color(YELLOW)
        )
        
        # Highlight the higher order terms
        powers_rect = SurroundingRectangle(matrix_series_v1, color=YELLOW, buff=0.1)
        self.play(Create(powers_rect))
        self.wait(1)
        self.play(FadeOut(powers_rect))

        # === Animation for Lecture Line 5 ===
        # "The resulting sum defines the matrix exponential."
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(YELLOW)
        )
        
        # Objects for final state per VideoCritic issues
        diff_eq = Text("d/dt x(t) = A x(t)", font_size=32, color=COLOR_EQ_BLUE)
        solution = Text("x(t) = e^(At) x(0)", font_size=32, color=COLOR_EQ_YELLOW)
        
        # Apply layout fixes from issues 33, 34, 35, 47
        self.place_at_grid(diff_eq, 'B4', scale_factor=0.8)
        self.place_at_grid(solution, 'C4', scale_factor=0.8)
        
        # Final expansion position: Line 72: self.place_in_area(series_expansion, 'D1', 'D6', scale_factor=0.85)
        # Note: we use matrix_series_v1 which is our series_expansion object
        self.play(
            scalar_series.animate.set_opacity(0.2),
            FadeIn(diff_eq),
            FadeIn(solution),
            self.place_in_area(matrix_series_v1, 'D1', 'D6', scale_factor=0.85).animate
        )
        
        final_box = SurroundingRectangle(matrix_series_v1, color=WHITE, buff=0.2)
        self.play(Create(final_box))
        self.wait(3)
