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

class Section7Scene(TeachingScene):
    def construct(self):
        # Fetching title and lecture lines from storyboard
        title = "Summary and Conclusion"
        lecture_lines = [
            "- Convolution uses a sliding window to transform data.",
            "- It extracts meaningful information from local neighborhoods.",
            "- From blurring to edge detection, the logic remains identical."
        ]
        
        self.setup_layout(title, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Display a kernel sliding across a generic 2D signal grid.
        self.lecture[0].set_color(YELLOW)
        
        # Create a 5x5 grid. Each cell is 0.6 units.
        grid_cells = VGroup(*[Square(side_length=0.6, stroke_width=1, color=GRAY) for _ in range(25)])
        grid_cells.arrange_in_grid(rows=5, cols=5, buff=0)
        # Fix for Issue 37: Move from 'A1' to 'A2' to avoid lecture text
        self.place_in_area(grid_cells, "A2", "E6", scale_factor=1.0)
        
        # 3x3 kernel highlighter
        kernel_rect = Square(side_length=1.8, stroke_width=4, color=YELLOW, fill_opacity=0.2, fill_color=YELLOW)
        
        # Center of top-left 3x3 area is grid_cells[6]
        kernel_rect.move_to(grid_cells[6].get_center())
        
        self.play(FadeIn(grid_cells))
        self.play(Create(kernel_rect))
        
        # Slide sequence: traversal of the 5x5 grid with a 3x3 kernel
        slide_indices = [7, 8, 13, 12, 11, 16, 17, 18]
        for idx in slide_indices:
            self.play(kernel_rect.animate.move_to(grid_cells[idx].get_center()), run_time=0.4, rate_func=linear)
            
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show three small windows: Original, Blurred, and Edge-detected results.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(BLUE)
        
        # Clear grid and kernel
        self.play(FadeOut(grid_cells), FadeOut(kernel_rect))
        
        # Mini visuals for results
        def create_result_box(label, color, pattern_type):
            box = Square(side_length=1.2, stroke_width=2, color=color)
            if pattern_type == "orig":
                # Grid of varying white dots (deterministic pattern for performance)
                content = VGroup(*[Dot(radius=0.04, color=WHITE, fill_opacity=0.3 + 0.1 * (i % 7)) for i in range(16)])
                content.arrange_in_grid(rows=4, cols=4, buff=0.15)
            elif pattern_type == "blur":
                # Large soft blue box
                content = Square(side_length=1.0, fill_opacity=0.4, fill_color=BLUE, stroke_width=0)
            else: # edge
                # Red outline
                content = Square(side_length=0.8, stroke_width=4, color=RED)
            
            lbl = Text(label, font_size=16, color=WHITE)
            res = VGroup(box, content, lbl).arrange(DOWN, buff=0.2)
            return res

        orig_box = create_result_box("Original", WHITE, "orig")
        blur_box = create_result_box("Blurred", BLUE, "blur")
        edge_box = create_result_box("Edges", RED, "edge")
        
        # Fix for Issue 38: Shift boxes to the right to avoid crowding lecture text
        self.place_at_grid(orig_box, "C2", scale_factor=0.9)
        self.place_at_grid(blur_box, "C4", scale_factor=0.9)
        self.place_at_grid(edge_box, "C6", scale_factor=0.9)
        
        self.play(FadeIn(orig_box), FadeIn(blur_box), FadeIn(edge_box))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Display final text 'Transforming Data through Local Neighborhoods' in #FFFFFF.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(WHITE)
        
        final_text = Text("Transforming Data through\nLocal Neighborhoods", font_size=26, color="#FFFFFF")
        # Fix for Issue 39: Move from 'E1' to 'E2' for alignment
        self.place_in_area(final_text, "E2", "F6", scale_factor=1.0)
        
        self.play(Write(final_text))
        self.wait(3)
