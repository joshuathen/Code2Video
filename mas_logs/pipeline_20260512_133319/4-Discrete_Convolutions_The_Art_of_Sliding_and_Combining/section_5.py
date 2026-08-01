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
        # Setup layout with title and lecture lines
        lecture_lines = [
            "Convolution extends naturally to two-dimensional image grids.",
            "A small kernel slides over pixels to find patterns.",
            "Specialized kernels highlight edges in digital images.",
            "This helps self-driving cars identify highway lane markings.",
            "Feature maps extract essential information for the computer."
        ]
        self.setup_layout("Application 2: Feature Detection in 2D", lecture_lines)

        # Helper to create a 2D grid of squares
        def create_pixel_grid(rows, cols, size=0.4, color="#888888"):
            grid = VGroup()
            for r in range(rows):
                for c in range(cols):
                    sq = Square(side_length=size, stroke_width=1, fill_opacity=0.3, fill_color=color, color=color)
                    sq.move_to(np.array([c * size, -r * size, 0]))
                    grid.add(sq)
            return grid

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        input_grid = create_pixel_grid(5, 5, size=0.5, color="#888888")
        # ISSUE 35 FIX: Moved to A2-D3 and scaled to 0.8
        self.place_in_area(input_grid, "A2", "D3", scale_factor=0.8)
        input_label = Text("Input Image (Pixels)", font_size=18, color=WHITE)
        input_label.next_to(input_grid, UP, buff=0.2)
        
        self.play(FadeIn(input_grid), FadeIn(input_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Sobel Kernel frame
        kernel_frame = VGroup()
        for r in range(3):
            for c in range(3):
                sq = Square(side_length=0.4, stroke_width=3, color="#FFFF00") # Scaled slightly with grid
                sq.move_to(np.array([c * 0.4, -r * 0.4, 0]))
                kernel_frame.add(sq)
        
        # Sobel X values text
        sobel_vals = [["-1", "0", "1"], ["-2", "0", "2"], ["-1", "0", "1"]]
        kernel_texts = VGroup()
        for r in range(3):
            for c in range(3):
                txt = Text(sobel_vals[r][c], font_size=12, color="#FFFF00")
                txt.move_to(kernel_frame[r*3 + c].get_center())
                kernel_texts.add(txt)
        
        kernel = VGroup(kernel_frame, kernel_texts)
        # Position kernel at top-left of input grid (accounting for grid scaling)
        # Square size in input_grid is 0.5 * 0.8 = 0.4
        kernel.move_to(input_grid[0].get_center() + np.array([0.4, -0.4, 0])) 
        
        self.play(FadeIn(kernel))
        self.wait(0.5)
        
        # Slide kernel to a few positions
        # Pos 1: (0,0) - already there
        # Pos 2: (0,1)
        self.play(kernel.animate.shift(RIGHT * 0.4), run_time=0.8)
        # Pos 3: (0,2)
        self.play(kernel.animate.shift(RIGHT * 0.4), run_time=0.8)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Highlight vertical edge (Column 3 of input grid)
        edge_indices = [2, 7, 12, 17, 22, 3, 8, 13, 18, 23] # Columns 2 and 3 (0-indexed)
        highlight_anims = []
        for idx in edge_indices:
            highlight_anims.append(input_grid[idx].animate.set_fill("#FFFFFF", opacity=0.8).set_color("#FFFFFF"))
        
        self.play(*highlight_anims)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Output grid (3x3)
        output_grid = create_pixel_grid(3, 3, size=0.5, color="#888888")
        # ISSUE 36 FIX: Moved to A4-D5 and scaled to 0.8
        self.place_in_area(output_grid, "A4", "D5", scale_factor=0.8)
        output_label = Text("Feature Map (Edges)", font_size=18, color=WHITE)
        output_label.next_to(output_grid, UP, buff=0.2)
        
        # Highlight center of output as edge detection
        output_grid[1].set_fill(WHITE, opacity=0.9)
        output_grid[4].set_fill(WHITE, opacity=0.9)
        output_grid[7].set_fill(WHITE, opacity=0.9)
        
        arrow = Arrow(input_grid.get_right(), output_grid.get_left(), color=WHITE, buff=0.1)
        
        self.play(FadeIn(output_grid), FadeIn(output_label), GrowArrow(arrow))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Clear grids and show car visual
        self.play(FadeOut(input_grid), FadeOut(input_label), FadeOut(output_grid), FadeOut(output_label), FadeOut(kernel), FadeOut(arrow))
        
        # Simple car lane detection visual
        # Lane lines
        left_lane = Line(self.grid["F2"], self.grid["B3"], color=WHITE, stroke_width=4)
        right_lane = Line(self.grid["F5"], self.grid["B4"], color=WHITE, stroke_width=4)
        
        # Road surface
        road = Polygon(self.grid["F2"], self.grid["B3"], self.grid["B4"], self.grid["F5"], fill_color="#333333", fill_opacity=0.5, stroke_width=0)
        
        # Simplified car (from above/rear)
        car_body = Rectangle(width=1.2, height=0.8, fill_color=BLUE, fill_opacity=0.8, color=WHITE)
        # ISSUE 37 FIX: Moved to area E3-E4 and scaled to 1.2. Removed manual shift.
        self.place_in_area(car_body, "E3", "E4", scale_factor=1.2)
        
        self.play(FadeIn(road), Create(left_lane), Create(right_lane))
        self.play(FadeIn(car_body))
        
        # Highlight lane lines as being "detected"
        flash_left = left_lane.copy().set_color(YELLOW).set_stroke(width=8)
        flash_right = right_lane.copy().set_color(YELLOW).set_stroke(width=8)
        
        self.play(FadeIn(flash_left), FadeIn(flash_right))
        self.play(FadeOut(flash_left), FadeOut(flash_right))
        
        self.wait(2)
