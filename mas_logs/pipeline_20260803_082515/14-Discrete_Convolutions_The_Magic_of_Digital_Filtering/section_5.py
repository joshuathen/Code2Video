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
        # Setup content
        title_text = "2D Convolution: How Images 'See'"
        lecture_lines = [
            "In 2D, the kernel slides over a pixel grid.",
            "Each output pixel is a weighted sum of neighbors.",
            "A blur kernel spreads colors to nearby pixels.",
            "This process is the foundation of modern computer vision.",
            "CNNs use many kernels to extract complex image features."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        COLOR_INPUT = "#E0E0E0"
        COLOR_KERNEL = "#FF0000"
        COLOR_OUTPUT = "#90EE90"
        COLOR_CV = "#ADD8E6"
        COLOR_CNN = "#FFFFE0"

        # Assets
        camera_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg"
        computer_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg"

        # Helper for pixel positions in a 5x5 grid with 0.4 sized cells
        def get_pixel_pos(grid, r, c):
            # r, c from 0 to 4. Grid is 5x5.
            return grid.get_center() + (c - 2) * 0.4 * RIGHT + (2 - r) * 0.4 * UP

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_INPUT)
        
        # Show a 5x5 grid of pixels #E0E0E0.
        input_grid = VGroup(*[
            Square(side_length=0.4, fill_opacity=0.3, fill_color=COLOR_INPUT, stroke_color=WHITE, stroke_width=1)
            for _ in range(25)
        ]).arrange_in_grid(5, 5, buff=0)
        self.place_in_area(input_grid, 'B1', 'D3')
        
        # [Asset: camera.svg] - camera icon at A1
        camera_icon = SVGMobject(camera_path, color=WHITE)
        self.place_at_grid(camera_icon, 'A1', scale_factor=0.3)
        
        # Improved label positioning per Issue 41
        input_label = Text("Input Image", font_size=20, color=WHITE)
        self.place_in_area(input_label, 'A1', 'A3', scale_factor=0.8)
        
        self.play(FadeIn(camera_icon))
        self.play(Create(input_grid), Write(input_label))
        
        # A 3x3 red frame #FF0000 slides over the 5x5 grid.
        red_frame = Square(side_length=1.2, stroke_color=COLOR_KERNEL, stroke_width=4)
        red_frame.move_to(get_pixel_pos(input_grid, 1, 1)) # Start at top-left-ish valid center
        
        self.play(Create(red_frame))
        self.play(red_frame.animate.move_to(get_pixel_pos(input_grid, 0, 0)), run_time=1)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(COLOR_KERNEL)
        
        # Each step populates a pixel in a new 5x5 output grid #90EE90.
        output_grid = VGroup(*[
            Square(side_length=0.4, fill_opacity=0.0, fill_color=COLOR_OUTPUT, stroke_color=WHITE, stroke_width=1)
            for _ in range(25)
        ]).arrange_in_grid(5, 5, buff=0)
        self.place_in_area(output_grid, 'B4', 'D6')
        
        # Improved label positioning per Issue 41
        output_label = Text("Output Grid", font_size=20, color=WHITE)
        self.place_in_area(output_label, 'A4', 'A6', scale_factor=0.8)
        
        self.play(Create(output_grid), Write(output_label))

        # Highlight box for neighbors
        neighbor_highlight = Square(side_length=1.2, fill_color=COLOR_KERNEL, fill_opacity=0.2, stroke_width=0)
        neighbor_highlight.move_to(red_frame.get_center())
        self.add(neighbor_highlight)
        
        def perform_conv(r, c, run_time=0.4):
            pos = get_pixel_pos(input_grid, r, c)
            self.play(
                red_frame.animate.move_to(pos),
                neighbor_highlight.animate.move_to(pos),
                run_time=run_time
            )
            target_pixel = output_grid[r*5 + c]
            self.play(
                target_pixel.animate.set_fill(opacity=0.8).set_stroke(opacity=1),
                run_time=run_time/2
            )

        # Sequence of convolution steps
        perform_conv(0, 0)
        perform_conv(0, 1)
        perform_conv(0, 2)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(COLOR_OUTPUT)
        # A blur kernel spreads colors to nearby pixels.
        perform_conv(1, 0)
        perform_conv(1, 1)
        perform_conv(1, 2)
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(COLOR_CV)
        # This process is the foundation of modern computer vision.
        
        # Fill the rest quickly
        rest_of_pixels = []
        for r in range(5):
            for c in range(5):
                idx = r*5 + c
                if output_grid[idx].get_fill_opacity() == 0:
                    rest_of_pixels.append(output_grid[idx].animate.set_fill(opacity=0.8).set_stroke(opacity=1))
        
        self.play(
            red_frame.animate.move_to(get_pixel_pos(input_grid, 4, 4)),
            neighbor_highlight.animate.move_to(get_pixel_pos(input_grid, 4, 4)),
            *rest_of_pixels,
            run_time=1.5
        )
        self.wait(0.5)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(COLOR_CNN)
        
        # [Asset: computer.svg] - computer icon at E1
        computer_icon = SVGMobject(computer_path, color=WHITE)
        self.place_at_grid(computer_icon, 'E1', scale_factor=0.3)
        
        # Mathematical formula
        formula = MathTex(
            r"\text{Output} = \sum (\text{Kernel} \times \text{Neighbors})", 
            color=COLOR_CNN,
            font_size=32
        )
        # Improved formula positioning per Issue 42
        self.place_in_area(formula, 'F1', 'F6', scale_factor=0.8)
        
        self.play(FadeIn(computer_icon))
        self.play(Write(formula))
        self.wait(2)
