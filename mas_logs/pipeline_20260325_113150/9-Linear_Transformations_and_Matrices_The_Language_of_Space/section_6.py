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

class Section6Scene(TeachingScene):
    def construct(self):
        # Setup layout with title and lecture lines
        title_text = "Application: Character Animation"
        lecture_lines = [
            "Matrices power the graphics in your favorite games.",
            "They can scale, rotate, or shear objects effortlessly.",
            "Watch this shape lean using a shear transformation."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Highlight first line in yellow
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        
        # Define a local coordinate system (NumberPlane)
        # We'll place it in the center of the right area (A1 to F6)
        coord_sys = NumberPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            background_line_style={"stroke_color": "#FFFFFF", "stroke_width": 1, "stroke_opacity": 0.2},
            axis_config={"stroke_color": "#FFFFFF", "stroke_width": 2}
        )
        # Adjust scale_factor to 0.6 to avoid screen edge clipping after transformation
        self.place_in_area(coord_sys, "A1", "F6", scale_factor=0.6)
        
        # Create a simple square shape (made of lines #FFFFFF)
        square = Square(side_length=1.4, color="#FFFFFF", stroke_width=4)
        # Define scale_factor to match coord_sys for alignment and size consistency
        self.place_in_area(square, "A1", "F6", scale_factor=0.6) 
        
        self.play(Create(coord_sys), Create(square))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second line in light blue
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#00BFFF")
        )
        
        # Demonstrate scale and rotate briefly as mentioned in the text
        # Scale up and down
        self.play(square.animate.scale(1.3), run_time=0.6)
        self.play(square.animate.scale(1/1.3), run_time=0.6)
        # Rotate slightly
        self.play(Rotate(square, angle=PI/4), run_time=0.7)
        self.play(Rotate(square, angle=-PI/4), run_time=0.7)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight third line in green to match the transformation result
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#00FF00")
        )
        
        # Define the shear matrix: [[1, 1], [0, 1]]
        shear_matrix = [[1, 1], [0, 1]]
        
        # Construct the matrix mobject manually using Text to avoid LaTeX dependency
        matrix_elements = VGroup(*[
            VGroup(*[Text(str(cell), font_size=30) for cell in row]).arrange(RIGHT, buff=0.5)
            for row in shear_matrix
        ]).arrange(DOWN, buff=0.3)
        
        l_bracket = Text("[", font_size=45).stretch_to_fit_height(matrix_elements.height + 0.1)
        l_bracket.next_to(matrix_elements, LEFT, buff=0.1)
        r_bracket = Text("]", font_size=45).stretch_to_fit_height(matrix_elements.height + 0.1)
        r_bracket.next_to(matrix_elements, RIGHT, buff=0.1)
        
        matrix_tex = VGroup(l_bracket, matrix_elements, r_bracket).scale(0.7).set_color("#00FF00")
        
        # Place label at grid position A3 for better centering and visibility
        self.place_at_grid(matrix_tex, "A3", scale_factor=0.8)
        
        self.play(Write(matrix_tex))
        
        # Apply the shear transformation to both the grid and the square simultaneously
        self.play(
            coord_sys.animate.apply_matrix(shear_matrix),
            square.animate.apply_matrix(shear_matrix),
            run_time=2
        )
        
        # Highlight the transformation's result (rhombus) by changing color to green
        self.play(
            square.animate.set_color("#00FF00"),
            square.animate.set_stroke(width=6)
        )
        
        self.wait(3)
