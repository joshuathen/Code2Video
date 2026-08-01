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

        # Define fine-grained animation grid (6x6 grid on right side)
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
        # 1. Setup
        title_str = "Decoding the Matrix: Where do the Basis Vectors land?"
        lines = [
            "- A matrix tells us where basis vectors land.",
            "- The first column shows i-hat's new position.",
            "- The second column shows j-hat's new position.",
            "- These two landing spots define the transformed space.",
            "- Numbers in the matrix now have a geometric meaning."
        ]
        self.setup_layout(title_str, lines)
        
        # Constants
        COLOR_I = "#FF0000"
        COLOR_J = "#00FF00"
        HIGHLIGHT_COLOR = "#FFFF00"
        
        # Asset Paths (kept from current code)
        matrix_asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/matrix.svg"
        grid_asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        
        # Matrix Setup: [[2, 1], [-1, 1]]
        matrix_vals = [[2, 1], [-1, 1]]
        matrix_mobject = Matrix(
            matrix_vals,
            left_bracket=None,
            right_bracket=None,
            element_to_mobject=lambda m: Text(str(m), font_size=24),
            element_alignment_corner=ORIGIN
        ).scale(0.7)
        
        # Add brackets manually using Text to avoid LaTeX dependency
        lb = Text("[", font_size=32).stretch_to_fit_height(matrix_mobject.height + 0.2).next_to(matrix_mobject, LEFT, buff=0.1)
        rb = Text("]", font_size=32).stretch_to_fit_height(matrix_mobject.height + 0.2).next_to(matrix_mobject, RIGHT, buff=0.1)
        matrix_mobject.add(lb, rb)
        
        # Color columns to match the basis vectors
        matrix_mobject.get_columns()[0].set_color(COLOR_I)
        matrix_mobject.get_columns()[1].set_color(COLOR_J)
        
        # Matrix SVG asset (handling potential missing file with a circle placeholder if needed, 
        # but following original logic)
        try:
            matrix_svg = SVGMobject(matrix_asset_path).scale(0.4)
        except:
            matrix_svg = Circle(radius=0.3, color=WHITE)
            
        matrix_group = VGroup(matrix_mobject, matrix_svg).arrange(RIGHT, buff=0.4)
        self.place_in_area(matrix_group, 'A4', 'B6', scale_factor=1.0)
        
        self.play(FadeIn(matrix_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_I)
        
        # Coordinate System Visualization
        origin_pos = self.grid['D3']
        try:
            grid_bg = SVGMobject(grid_asset_path).scale(1.5).move_to(self.grid['D4'])
        except:
            grid_bg = NumberPlane(x_range=[0, 4], y_range=[-2, 2]).scale(0.5).move_to(self.grid['D4'])
        
        self.play(FadeIn(grid_bg))
        
        # Basis vector i-hat moves to (2,-1)
        i_end_pos = self.grid['E5']   # (2,-1) relative to D3
        i_vec = Arrow(origin_pos, i_end_pos, color=COLOR_I, buff=0)
        i_label = Text("i'", font_size=18, color=COLOR_I).next_to(i_end_pos, DOWN, buff=0.1)
        
        self.play(Create(i_vec), Write(i_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_J)
        
        # Basis vector j-hat moves to (1,1)
        j_end_pos = self.grid['C4']   # (1,1) relative to D3
        j_vec = Arrow(origin_pos, j_end_pos, color=COLOR_J, buff=0)
        j_label = Text("j'", font_size=18, color=COLOR_J).next_to(j_end_pos, UP, buff=0.1)
        
        self.play(Create(j_vec), Write(j_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(HIGHLIGHT_COLOR)
        
        # Parallelogram showing transformed area
        transformed_area = Polygon(
            origin_pos, 
            i_end_pos, 
            i_end_pos + (j_end_pos - origin_pos), 
            j_end_pos, 
            fill_opacity=0.3, 
            fill_color=BLUE, 
            stroke_width=1
        )
        
        self.play(FadeIn(transformed_area))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(HIGHLIGHT_COLOR)
        
        # Final highlight on matrix numbers
        highlight_box = SurroundingRectangle(matrix_mobject, color=HIGHLIGHT_COLOR)
        self.play(Create(highlight_box))
        self.wait(2)

        # Clean up
        self.play(FadeOut(highlight_box), FadeOut(transformed_area), FadeOut(i_vec), FadeOut(j_vec), FadeOut(i_label), FadeOut(j_label), FadeOut(matrix_group), FadeOut(grid_bg))
        self.lecture[4].set_color(WHITE)
        self.wait(1)