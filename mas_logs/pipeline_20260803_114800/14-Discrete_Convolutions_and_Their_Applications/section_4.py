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
        # Setup layout
        title_text = "Expanding to 2D: The Image Matrix"
        lecture_lines = [
            "- Images are matrices where each cell is a pixel.",
            "- 2D convolution uses a small square called a kernel.",
            "- The kernel slides across the image, row by row.",
            "- Multiply the kernel with the underlying pixel grid.",
            "- These local sums create a new, transformed image."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Seed for reproducibility
        np.random.seed(42)

        # Helper to create a grid-based matrix
        def create_matrix(rows, cols, cell_size=0.6, color=WHITE, add_values=True, font_size=14):
            matrix = VGroup()
            for r in range(rows):
                for c in range(cols):
                    sq = Square(side_length=cell_size, color=color, stroke_width=2)
                    # Position relative to top-left of matrix
                    sq.move_to(r * DOWN * cell_size + c * RIGHT * cell_size)
                    matrix.add(sq)
                    if add_values:
                        val = np.random.randint(10, 99)
                        txt = Text(str(val), font_size=font_size, color=color).move_to(sq.get_center())
                        matrix.add(txt)
            matrix.move_to(ORIGIN)
            return matrix

        # === Animation for Lecture Line 1 ===
        # Images are matrices where each cell is a pixel.
        self.lecture[0].set_color(BLUE_C)
        input_matrix = create_matrix(5, 5, color=BLUE_C)
        # Resolved Issue 26: Moving matrix further right to avoid lecture overlap
        self.place_in_area(input_matrix, "A3", "E5")
        self.play(Create(input_matrix))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # 2D convolution uses a small square called a kernel.
        self.lecture[1].set_color("#FFFF00") # Bright Yellow
        kernel = create_matrix(3, 3, color="#FFFF00", add_values=False)
        # Identity kernel values (1 in center, 0 elsewhere)
        kernel_data = [0, 0, 0, 0, 1, 0, 0, 0, 0]
        kernel_texts = VGroup()
        for i, val in enumerate(kernel_data):
            t = Text(str(val), font_size=16, color="#FFFF00").move_to(kernel[i].get_center())
            kernel_texts.add(t)
        kernel.add(kernel_texts)
        
        # Resolved Issue 28: Positioning kernel at A6 for better top-down flow
        self.place_at_grid(kernel, "A6", scale_factor=0.8)
        self.play(Create(kernel))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The kernel slides across the image, row by row.
        self.lecture[2].set_color(ORANGE)
        
        # Extract squares and texts for indexing from the 5x5 input matrix
        # Input matrix structure: 25 squares, then their 25 texts
        # Correction: create_matrix adds them in order: sq1, txt1, sq2, txt2...
        in_squares = VGroup(*[input_matrix[i*2] for i in range(25)])
        in_texts = VGroup(*[input_matrix[i*2+1] for i in range(25)])
        
        # Place kernel on the top-left 3x3 of the input
        # Center of 3x3 subgrid (0,0)-(2,2) is the center of input cell (1,1)
        start_pos = in_squares[1*5 + 1].get_center()
        self.play(kernel.animate.move_to(start_pos).scale(1.25)) # Return to scale 1.0 (0.8 * 1.25)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Multiply the kernel with the underlying pixel grid.
        self.lecture[3].set_color(PINK)
        
        # Create output matrix (3x3)
        output_matrix = create_matrix(3, 3, color="#00FFFF", add_values=False)
        # Resolved Issue 27: Positioning output_matrix at E5-F6 to reduce cramping
        self.place_in_area(output_matrix, "E5", "F6")
        out_squares = VGroup(*[output_matrix[i] for i in range(9)])
        self.play(Create(out_squares))
        
        out_texts = VGroup()
        
        # Step through the sliding convolution (3x3 output for 3x3 kernel on 5x5 input)
        idx = 0
        for r in range(3):
            for c in range(3):
                # Target center is cell (r+1, c+1) of the input matrix
                target_center = in_squares[(r+1)*5 + (c+1)].get_center()
                
                if idx > 0:
                    # Move kernel to next position
                    self.play(kernel.animate.move_to(target_center), run_time=0.4)
                
                # Using identity kernel: picking the center pixel value
                val_str = in_texts[(r+1)*5 + (c+1)].text
                out_t = Text(val_str, font_size=16, color="#00FFFF").move_to(out_squares[idx].get_center())
                
                # Highlighting result calculation
                self.play(Write(out_t), run_time=0.2)
                out_texts.add(out_t)
                idx += 1
        
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # These local sums create a new, transformed image.
        self.lecture[4].set_color("#00FFFF") # Cyan
        
        # Final highlight of the transformation
        full_output = VGroup(out_squares, out_texts)
        self.play(
            full_output.animate.scale(1.1),
            kernel.animate.set_stroke(opacity=0.3),
            input_matrix.animate.set_stroke(opacity=0.3)
        )
        self.wait(2)
