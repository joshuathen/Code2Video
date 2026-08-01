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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("2D Application: Digital Image Filters", [
            "Images are two-dimensional grids of intensity values.",
            "A small matrix, called a kernel, slides over pixels.",
            "Different kernel values produce blur, sharpen, or edge effects.",
            "Edge detectors highlight rapid changes in pixel brightness.",
            "These 2D convolutions are the core of modern AI."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Images are two-dimensional grids of intensity values.
        self.lecture[0].set_color(YELLOW)
        
        # Create 5x5 Input Grid
        input_vals = [
            [10, 20, 30, 40, 50],
            [10, 200, 200, 200, 50],
            [10, 200, 255, 200, 50],
            [10, 200, 200, 200, 50],
            [10, 20, 30, 40, 50]
        ]
        
        input_grid = VGroup()
        for r in range(5):
            for c in range(5):
                val = input_vals[r][c]
                square = Square(side_length=0.5, stroke_width=2, color=GRAY)
                # Map intensity to color (0=black, 255=white)
                square.set_fill(interpolate_color(BLACK, WHITE, val/255.0), opacity=0.8)
                label = Text(str(val), font_size=14).move_to(square.get_center())
                cell = VGroup(square, label)
                cell.shift(r * 0.5 * DOWN + c * 0.5 * RIGHT)
                input_grid.add(cell)
        
        # Fix Issue 35: Better utilization of vertical space
        self.place_in_area(input_grid, "B1", "F3", scale_factor=0.9)
        
        # Fix Issue 37: Precise anchoring for labels
        input_title = Text("Input Image", font_size=18, color=YELLOW)
        self.place_at_grid(input_title, "A2", scale_factor=1.0)
        
        self.play(FadeIn(input_grid), Write(input_title))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # A small matrix, called a kernel, slides over pixels.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Create 3x3 Yellow Frame (Kernel)
        # Kernel center starts at (1,1) in the 5x5 grid (index 6)
        kernel_frame = Rectangle(
            width=input_grid[0].width * 3, 
            height=input_grid[0].height * 3, 
            stroke_width=4, 
            color=YELLOW
        ).move_to(input_grid[6].get_center()) 
        
        # Output Grid (3x3)
        output_grid = VGroup()
        for r in range(3):
            for c in range(3):
                square = Square(side_length=0.5, stroke_width=2, color=GRAY)
                square.set_fill(BLACK, opacity=1.0)
                square.shift(r * 0.5 * DOWN + c * 0.5 * RIGHT)
                output_grid.add(square)
        
        # Fix Issue 36: Visual consistency and centering
        self.place_in_area(output_grid, "C4", "E6", scale_factor=0.9)
        
        # Fix Issue 37: Precise anchoring for labels
        output_title = Text("Output Grid", font_size=18, color=YELLOW)
        self.place_at_grid(output_title, "B5", scale_factor=1.0)
        
        self.play(
            Create(kernel_frame), 
            FadeIn(output_grid),
            Write(output_title)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Different kernel values produce blur, sharpen, or edge effects.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Sliding Animation
        # Indices of center pixels for 3x3 window in 5x5 grid:
        # (1,1), (1,2), (1,3), (2,1), (2,2), (2,3), (3,1), (3,2), (3,3)
        center_indices = [6, 7, 8, 11, 12, 13, 16, 17, 18]
        
        # Visual simulation of edge detection (sharper contrast in output)
        output_colors = [
            GRAY_B, WHITE, GRAY_B,
            WHITE, BLACK, WHITE,
            GRAY_B, WHITE, GRAY_B
        ]

        for i, idx in enumerate(center_indices):
            self.play(
                kernel_frame.animate.move_to(input_grid[idx].get_center()),
                output_grid[i].animate.set_fill(output_colors[i]),
                run_time=0.4
            )
        
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Edge detectors highlight rapid changes in pixel brightness.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Edge highlight on input
        # Focus on the high contrast boundaries
        edge_indices = [6, 7, 8, 11, 13, 16, 17, 18]
        edge_highlight = VGroup(*[input_grid[i][0] for i in edge_indices])
        
        self.play(edge_highlight.animate.set_stroke(YELLOW, width=4))
        self.wait(1)
        self.play(edge_highlight.animate.set_stroke(GRAY, width=2))

        # === Animation for Lecture Line 5 ===
        # These 2D convolutions are the core of modern AI.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        self.play(Indicate(output_grid, color=YELLOW))
        self.wait(2)
