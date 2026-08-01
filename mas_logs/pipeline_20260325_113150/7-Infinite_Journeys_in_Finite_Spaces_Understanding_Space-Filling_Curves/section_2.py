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
        # Section title and lecture lines
        title_text = "Prerequisite: Iterative Functions and Limits"
        lecture_lines = [
            "Complex shapes often emerge from simple, repeated rules.",
            "We divide a square into a smaller, recursive grid.",
            "Repeating this process infinitely creates the final structure."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Display a simple square outline in #FFFFFF
        # Based on 6x6 grid, B2 to E5 defines a 3x3 grid unit square area
        main_square = Square(side_length=3.0, color="#FFFFFF")
        # Place in area spanning B2 to E5 (central on the right side)
        self.place_in_area(main_square, "B2", "E5")
        
        self.play(Create(main_square))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )

        def get_grid(n, size=3.0, color="#555555"):
            grid_lines = VGroup()
            step = size / n
            # Start from the bottom-left corner of the positioned square
            start_pos = main_square.get_corner(DL)
            
            # Vertical subdivision lines
            for i in range(1, n):
                v_line = Line(
                    start_pos + RIGHT * i * step,
                    start_pos + RIGHT * i * step + UP * size,
                    color=color,
                    stroke_width=1
                )
                grid_lines.add(v_line)
                
            # Horizontal subdivision lines
            for i in range(1, n):
                h_line = Line(
                    start_pos + UP * i * step,
                    start_pos + UP * i * step + RIGHT * size,
                    color=color,
                    stroke_width=1
                )
                grid_lines.add(h_line)
            return grid_lines

        # Divide square into 2x2
        grid_2x2 = get_grid(2)
        self.play(Create(grid_2x2))
        self.wait(0.5)
        
        # Subdivide each cell into another 2x2 grid (results in 4x4)
        grid_4x4 = get_grid(4)
        self.play(ReplacementTransform(grid_2x2, grid_4x4))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Rapidly animate the grid refining from 4x4 to 8x8 to 16x16 cells
        grid_8x8 = get_grid(8)
        grid_16x16 = get_grid(16)
        
        self.play(ReplacementTransform(grid_4x4, grid_8x8), run_time=0.6)
        self.play(ReplacementTransform(grid_8x8, grid_16x16), run_time=0.4)
        
        self.wait(3)
