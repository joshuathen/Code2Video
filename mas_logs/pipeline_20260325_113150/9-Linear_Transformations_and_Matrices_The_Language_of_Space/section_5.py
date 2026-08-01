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
        # 1. Setup layout with mandatory title and lecture lines
        lecture_lines = [
            "Multiply any vector by the matrix for its transformation.",
            "Scale the new basis vectors by the vector's coordinates.",
            "Compute the new x-coordinate by summing these components.",
            "Compute the new y-coordinate in the same way.",
            "The vector then jumps to its final calculated position."
        ]
        self.setup_layout("Calculation: Moving any Vector", lecture_lines)

        # Define colors for consistency and clarity
        I_COLOR = "#88FF88" # Light Green
        J_COLOR = "#FF8888" # Light Red
        V_COLOR = "#8888FF" # Light Blue
        RES_COLOR = "#FFFF88" # Light Yellow

        # === Animation for Lecture Line 1 ===
        # Highlight first line
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Matrix multiplication text (using Text/Monospace for reliability)
        # Matrix: [[3, 2], [-2, 1]], Vector: [1, 2]
        matrix_tex = Text("[[3, 2], [-2, 1]] * [1, 2]", font="Monospace", font_size=24)
        # Issue 37 Fix: Positioned at A4 for better spacing
        self.place_at_grid(matrix_tex, 'A4', scale_factor=0.8)
        self.play(Write(matrix_tex))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        # Decomposed form: 1 * [3, -2] + 2 * [2, 1]
        decomp_tex = Text("1 * [3, -2] + 2 * [2, 1]", font="Monospace", font_size=24)
        self.place_at_grid(decomp_tex, 'A4', scale_factor=0.8)
        self.play(ReplacementTransform(matrix_tex, decomp_tex))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(I_COLOR)
        )
        # Calculate x: (1*3) + (2*2) = 7
        x_calc = Text("x: (1*3) + (2*2) = 7", font="Monospace", font_size=20, color=I_COLOR)
        self.place_at_grid(x_calc, 'A5', scale_factor=0.8)
        self.play(Write(x_calc))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(J_COLOR)
        )
        # Calculate y: (1*-2) + (2*1) = 0
        y_calc = Text("y: (1*-2) + (2*1) = 0", font="Monospace", font_size=20, color=J_COLOR)
        self.place_at_grid(y_calc, 'A6', scale_factor=0.8)
        self.play(Write(y_calc))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(YELLOW)
        )
        
        # Issue 38/39 Fix: Place plane in area B2 to F6 to avoid crowding Row A
        plane = NumberPlane(
            x_range=[-1, 8, 1],
            y_range=[-3, 3, 1],
            x_length=4,
            y_length=4,
            background_line_style={"stroke_opacity": 0.3}
        )
        self.place_in_area(plane, 'B2', 'F6', scale_factor=0.9)

        # Vector [1, 2] on original grid
        vec = Arrow(plane.c2p(0, 0), plane.c2p(1, 2), buff=0, color=V_COLOR)
        vec_label = Text("[1, 2]", font_size=16, color=V_COLOR).next_to(vec.get_end(), UP, buff=0.1)

        self.play(Create(plane))
        self.play(GrowArrow(vec), Write(vec_label))
        self.wait(0.5)

        # Transformation: Move vector to [7, 0] and warp the plane
        new_vec_end = plane.c2p(7, 0)
        res_label = Text("[7, 0]", font_size=16, color=RES_COLOR)
        
        self.play(
            plane.animate.apply_matrix([[3, 2], [-2, 1]]),
            vec.animate.put_start_and_end_on(plane.c2p(0, 0), new_vec_end),
            vec_label.animate.move_to(new_vec_end + UP * 0.3).set_color(RES_COLOR),
            Transform(vec_label, res_label),
            run_time=2
        )
        
        # Cleanup highlighting
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(2)
