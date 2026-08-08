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
        # Setup layout with title and lecture lines
        self.setup_layout("Application: Digital Image Filtering", [
            "Image convolution uses a two-dimensional sliding matrix.",
            "These filters can blur, sharpen, or detect sharp edges.",
            "Convolution enables computers to \"see\" outlines in images."
        ])
        
        # Colors
        input_grid_color = GREY_D
        kernel_color = "#0000FF" # Blue
        edge_color = "#FFFFFF"   # White
        highlight_color = YELLOW
        text_flash_color = "#00FF00" # Green

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(highlight_color)
        
        # Create 6x6 pixel grid (Input)
        pixel_size = 0.15
        input_grid = VGroup(*[
            Square(side_length=pixel_size, stroke_width=1, stroke_color=GREY, fill_opacity=0.2, fill_color=input_grid_color)
            for _ in range(36)
        ]).arrange_in_grid(6, 6, buff=0)
        
        # Use scale_factor=1.3 for input_grid (Issue 32)
        self.place_at_grid(input_grid, "C2", scale_factor=1.3)
        
        input_label = Text("Input Image", font_size=16).next_to(input_grid, UP, buff=0.1)
        
        # 3x3 Blue Kernel Frame
        scaled_pixel_size = pixel_size * 1.3
        kernel = Rectangle(
            width=scaled_pixel_size * 3, 
            height=scaled_pixel_size * 3, 
            stroke_width=4, 
            stroke_color=kernel_color, 
            fill_opacity=0.3, 
            fill_color=kernel_color
        )
        
        # Initial kernel position: top-left 3x3 area
        start_pos = input_grid[0].get_center() + RIGHT * scaled_pixel_size + DOWN * scaled_pixel_size
        kernel.move_to(start_pos)
        
        self.play(FadeIn(input_grid), FadeIn(input_label))
        self.play(Create(kernel))
        
        # Demonstrate sliding horizontally
        for i in range(3):
            self.play(kernel.animate.shift(RIGHT * scaled_pixel_size), run_time=0.4)
        
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(highlight_color)
        
        # Create second 6x6 grid (Output)
        output_grid = VGroup(*[
            Square(side_length=pixel_size, stroke_width=1, stroke_color=GREY, fill_opacity=0.1, fill_color=BLACK)
            for _ in range(36)
        ]).arrange_in_grid(6, 6, buff=0)
        
        # Use scale_factor=1.3 for output_grid (Issue 33)
        self.place_at_grid(output_grid, "C5", scale_factor=1.3)
        output_label = Text("Filtered Output", font_size=16).next_to(output_grid, UP, buff=0.1)
        
        self.play(FadeIn(output_grid), FadeIn(output_label))
        
        # Reset kernel to top-left for the scan
        self.play(kernel.animate.move_to(start_pos), run_time=0.5)
        
        # Define indices for edges (a simple box shape) to visualize "detection"
        edge_centers = [
            (0, 1), (0, 2), 
            (1, 0), (1, 3),
            (2, 0), (2, 3),
            (3, 1), (3, 2)
        ]
        
        # Scan through 4x4 possible center positions
        for r in range(4):
            # Move kernel to start of the current row
            row_start = start_pos + DOWN * r * scaled_pixel_size
            self.play(kernel.animate.move_to(row_start), run_time=0.2)
            
            for c in range(1, 4): # We are already at c=0
                # Move kernel within the row
                pos = row_start + RIGHT * c * scaled_pixel_size
                self.play(kernel.animate.move_to(pos), run_time=0.15)
                
                # Check if this kernel position produces an "edge" pixel
                if (r, c) in edge_centers:
                    out_idx = (r + 1) * 6 + (c + 1)
                    self.play(output_grid[out_idx].animate.set_fill(edge_color, opacity=1), run_time=0.05)
        
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(highlight_color)
        
        edge_detection_text = Text("Edge Detection", color=text_flash_color, font_size=28)
        # Use place_in_area('E3', 'E4') for edge_detection_text (Issue 31)
        self.place_in_area(edge_detection_text, "E3", "E4", scale_factor=1.2)
        
        self.play(Flash(edge_detection_text, color=text_flash_color, line_length=0.3))
        self.play(FadeIn(edge_detection_text, shift=UP*0.2))
        
        # Visual flourish: highlight the output grid edges
        self.play(output_grid.animate.set_stroke(edge_color, width=2), run_time=1)
        
        self.wait(3)
