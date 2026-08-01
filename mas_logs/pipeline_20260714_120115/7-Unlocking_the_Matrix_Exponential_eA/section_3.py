from manim import *

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

class Section3Scene(Scene):
    def construct(self):
        # Title of the section
        title = Text("Matrix Exponential of a Diagonal Matrix", font_size=36)
        title.to_edge(UP)

        # Define matrix entries using characters that render well with Text
        # We avoid LaTeX-specific symbols like \lambda and \dots to prevent issues with Text rendering
        matrix_entries = [
            ["e^(λ1 t)", "0", "...", "0"],
            ["0", "e^(λ2 t)", "...", "0"],
            ["...", "...", "...", "..."],
            ["0", "0", "...", "e^(λn t)"]
        ]
        
        # Manually construct the matrix using VGroup and Text.
        # This bypasses the Matrix class, which internally calls MathTex to render brackets,
        # which is the source of the FileNotFoundError: 'latex'.
        
        # 1. Create the grid of elements
        grid_elements = VGroup()
        for row_data in matrix_entries:
            row_mobs = VGroup(*[Text(item, font_size=22) for item in row_data])
            row_mobs.arrange(RIGHT, buff=1.1)
            grid_elements.add(row_mobs)
        grid_elements.arrange(DOWN, buff=0.7)

        # 2. Create the brackets using Text mobjects
        # We stretch the text characters "[" and "]" to fit the height of our grid
        bracket_h = grid_elements.height + 0.4
        l_bracket = Text("[", font_size=40).stretch_to_fit_height(bracket_h)
        l_bracket.next_to(grid_elements, LEFT, buff=0.2)
        
        r_bracket = Text("]", font_size=40).stretch_to_fit_height(bracket_h)
        r_bracket.next_to(grid_elements, RIGHT, buff=0.2)
        
        # Group the grid and brackets into a single matrix object
        matrix = VGroup(grid_elements, l_bracket, r_bracket)
        
        # Define the left hand side of the equation using Text
        lhs = Text("e^(At) =", font_size=32)
        
        # Group and arrange the equation components
        equation = VGroup(lhs, matrix).arrange(RIGHT, buff=0.5)
        equation.next_to(title, DOWN, buff=1)

        # Animation sequence
        self.play(
            Write(title),
            run_time=1.5
        )
        self.wait(0.5)

        # Animate the construction of the equation
        self.play(
            Write(lhs),
            Create(l_bracket),
            Create(r_bracket),
            FadeIn(grid_elements, lag_ratio=0.1),
            run_time=2
        )
        self.wait(2)
