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
        # Initial Setup
        title = "Orientation and Negative Determinants"
        # Script-aligned lecture lines
        lines = [
            "Normally, i-hat is to the right of j-hat.",
            "Some transformations flip space over, reversing their relative order.",
            "A negative determinant indicates this change in orientation."
        ]
        self.setup_layout(title, lines)

        # Colors for lecture highlighting
        c1 = "#66CCFF"  # Light Blue
        c2 = "#FFEE88"  # Pale Yellow
        c3 = "#FF5555"  # Light Red

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(c1))

        # Create Coordinate Plane in the right area
        plane = NumberPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            x_length=3.5,
            y_length=3.5,
            background_line_style={"stroke_opacity": 0.4}
        )
        self.place_in_area(plane, 'B2', 'E5')

        # Create basis vectors
        i_vec = Arrow(plane.c2p(0, 0), plane.c2p(1, 0), buff=0, color=YELLOW, stroke_width=5)
        j_vec = Arrow(plane.c2p(0, 0), plane.c2p(0, 1), buff=0, color=GREEN, stroke_width=5)
        
        # Labels for basis vectors
        i_lab = Text("i", slant=ITALIC, color=YELLOW, font_size=24)
        j_lab = Text("j", slant=ITALIC, color=GREEN, font_size=24)

        # Position labels at grid markers relative to plane center
        # Issue 45: Applied scale_factor=0.8 to i_lab. 
        # Kept D5 for the initial "Standard" state (i to the right).
        self.place_at_grid(i_lab, 'D5', scale_factor=0.8) 
        self.place_at_grid(j_lab, 'C4', scale_factor=0.8)

        self.play(Create(plane), run_time=1)
        self.play(
            GrowArrow(i_vec), 
            GrowArrow(j_vec), 
            FadeIn(i_lab), 
            FadeIn(j_lab)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(c2))

        # Flipped orientation: Transform i-vector to (-1, 0)
        # This reflects the plane across the y-axis, resulting in det = -1
        # Issue 45: Move label to D2 to properly follow the flipped vector
        self.play(
            i_vec.animate.put_start_and_end_on(plane.c2p(0, 0), plane.c2p(-1, 0)),
            i_lab.animate.move_to(self.grid['D2']),
            run_time=2,
            rate_func=smooth
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(c3))

        # Final concluding text at the bottom of the right side
        concl_text = Text("Negative Determinant = Flipped Orientation", color=c3, font_size=24)
        # Issue 44: Reduced scale_factor to 0.6 to prevent clipping
        self.place_in_area(concl_text, 'F1', 'F6', scale_factor=0.6)

        self.play(Write(concl_text))
        self.wait(3)
