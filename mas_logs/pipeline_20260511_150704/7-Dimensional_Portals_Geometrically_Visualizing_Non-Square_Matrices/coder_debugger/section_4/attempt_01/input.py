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

class Section4Scene(TeachingScene):
    def construct(self):
        # Initial Setup
        title_text = "The Rule of Rows and Columns"
        lecture_lines = [
            "Columns represent the number of input basis vectors.",
            "Rows represent the coordinates in the output space.",
            "This simple rule governs all dimensional transformations."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Pre-create elements for logic
        # Matrix setup
        matrix_vals = [[ "a", "b" ], [ "c", "d" ], [ "e", "f" ]]
        matrix = Matrix(matrix_vals, left_bracket="[", right_bracket="]").scale(0.8)
        self.place_in_area(matrix, "A3", "C4", scale_factor=1.0)
        
        col_label = Text("Columns", font_size=20, color="#00FF00")
        row_label = Text("Rows", font_size=20, color="#FF0000")
        
        self.place_at_grid(col_label, "A3", scale_factor=1.0)
        col_label.shift(UP * 0.3)
        self.place_at_grid(row_label, "B2", scale_factor=1.0)
        row_label.shift(LEFT * 0.4)

        teal_circle = Circle(radius=0.3, color="#58C4DD", fill_opacity=0.8)
        red_square = Square(side_length=0.6, color="#E27367", fill_opacity=0.8)
        yellow_star = Star(n=5, color=YELLOW, fill_opacity=0.8)

        # Asset loading
        grid_asset = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/grid.svg")
        self.place_at_grid(grid_asset, "D4", scale_factor=0.6)

        # Summary text
        summary_text = Text("Columns = Input Dim, Rows = Output Dim", font_size=22, color=WHITE)
        self.place_in_area(summary_text, "F2", "F5", scale_factor=1.0)

        # === Animation for Lecture Line 1 ===
        # Columns: #00FF00
        self.play(self.lecture[0].animate.set_color("#00FF00"), run_time=0.5)
        self.play(Write(matrix), FadeIn(col_label))
        
        # Highlight matrix columns
        col_box = SurroundingRectangle(matrix.get_columns(), color="#00FF00", buff=0.1)
        self.place_at_grid(teal_circle, "B2", scale_factor=0.8) # Issue 38 fix
        
        self.play(
            Create(col_box),
            FadeIn(teal_circle)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Rows: #FF0000
        self.play(self.lecture[1].animate.set_color("#FF0000"), run_time=0.5)
        self.play(FadeIn(row_label))
        
        # Highlight matrix rows
        row_box = SurroundingRectangle(matrix.get_rows(), color="#FF0000", buff=0.1)
        self.place_at_grid(red_square, "D2", scale_factor=0.8) # Issue 39 fix
        
        self.play(
            ReplacementTransform(col_box, row_box),
            FadeIn(red_square)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Dimensional Portals (100x2)
        self.play(self.lecture[2].animate.set_color(WHITE), run_time=0.5)
        
        matrix_100x2 = MathTex(r"100 \times 2", font_size=36, color=WHITE)
        self.place_at_grid(matrix_100x2, "C4", scale_factor=1.0)
        
        # Transitioning to higher dim view
        self.play(
            FadeOut(matrix), 
            FadeOut(row_box), 
            FadeOut(row_label), 
            FadeOut(col_label),
            Write(matrix_100x2)
        )
        
        # Show Asset and "embedding"
        self.play(FadeIn(grid_asset))
        
        # Create "100 dimensions" representation (dense set of lines)
        dense_lines = VGroup(*[Line(LEFT, RIGHT, color=BLUE_E, stroke_width=0.5).shift(UP * i * 0.05) for i in range(-20, 20)])
        self.place_at_grid(dense_lines, "E4", scale_factor=0.8)
        
        self.place_at_grid(yellow_star, "C5", scale_factor=0.9) # Issue 37 fix
        
        self.play(
            grid_asset.animate.move_to(dense_lines.get_center()).set_opacity(0.3),
            FadeIn(dense_lines),
            FadeIn(yellow_star)
        )
        
        self.play(Write(summary_text))
        self.wait(3)
