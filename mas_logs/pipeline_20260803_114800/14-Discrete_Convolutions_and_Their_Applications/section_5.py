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
        self.setup_layout(
            "Application: Pixel-Bot’s Vision Enhancement", 
            [
                "Pixel-Bot uses Gaussian kernels to smooth out noisy images.",
                "Sobel kernels then highlight edges to detect doors and obstacles.",
                "Convolution transforms raw pixels into meaningful visual information."
            ]
        )
        
        # Define colors for elements and matching lecture lines
        BOT_COLOR = "#87CEFA"
        EDGE_COLOR = "#FF0000"
        BLUR_COLOR = "#FFFFE0" # Light Yellow for Blur highlight
        SUCCESS_COLOR = "#90EE90" # Light Green for result
        
        # === Animation for Lecture Line 1 ===
        # Line 1 Highlight
        self.play(self.lecture[0].animate.set_color(BOT_COLOR))
        
        # 1. Show 'Pixel-Bot' icon [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg]
        pixel_bot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        pixel_bot.set_color(BOT_COLOR)
        # Fix for issue 31: Place at B2
        self.place_at_grid(pixel_bot, "B2", scale_factor=0.8)
        bot_label = Text("Pixel-Bot", font_size=16, color=BOT_COLOR).next_to(pixel_bot, DOWN, buff=0.1)
        
        # 1.2 Noisy Image Grid (5x5)
        image_size = 5
        pixel_side = 0.4
        squares = VGroup()
        np.random.seed(42)
        for r in range(image_size):
            for c in range(image_size):
                shade = np.random.uniform(0.2, 0.8)
                sq = Square(side_length=pixel_side, fill_opacity=1, fill_color=interpolate_color(BLACK, WHITE, shade), stroke_width=0.5, stroke_color=GRAY_B)
                sq.move_to(RIGHT * (c * pixel_side) + DOWN * (r * pixel_side))
                squares.add(sq)
        
        # Fix for issue 30: Place in area B3 to E6
        self.place_in_area(squares, "B3", "E6", scale_factor=1.0)
        img_label = Text("Noisy Image", font_size=18, color=WHITE).next_to(squares, UP, buff=0.3)
        
        self.play(FadeIn(pixel_bot), FadeIn(bot_label))
        self.play(Create(squares), FadeIn(img_label))
        
        # 1.3 Gaussian Blur animation
        kernel_frame = Square(side_length=pixel_side * 2.8, color=BLUR_COLOR, stroke_width=4)
        # Position kernel at top-left of image (covering 3x3)
        kernel_frame.move_to(squares[6].get_center()) 
        kernel_name = Text("Gaussian Kernel", font_size=14, color=BLUR_COLOR).next_to(kernel_frame, UP, buff=0.05)
        
        self.play(Create(kernel_frame), FadeIn(kernel_name))
        self.wait(0.5)
        
        # Smooth image as kernel slides
        smoothed_gray = interpolate_color(BLACK, WHITE, 0.5)
        self.play(
            kernel_frame.animate.move_to(squares[18].get_center()), # Move to bottom-right area
            *[sq.animate.set_fill(smoothed_gray) for sq in squares],
            run_time=2
        )
        self.play(FadeOut(kernel_name))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line 2 Highlight
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(EDGE_COLOR)
        )
        
        # 2. Overlay a 'Sobel Edge' kernel
        sobel_name = Text("Sobel Kernel", font_size=14, color=EDGE_COLOR).next_to(kernel_frame, UP, buff=0.05)
        self.play(
            kernel_frame.animate.set_color(EDGE_COLOR),
            FadeIn(sobel_name)
        )
        
        # Edge Detection Logic (Highlighting boundary pixels)
        edge_indices = []
        for i in range(25):
            r, c = divmod(i, 5)
            if r == 0 or r == 4 or c == 0 or c == 4:
                edge_indices.append(i)
        
        self.play(
            kernel_frame.animate.move_to(squares[12].get_center()), # Move to center
            *[squares[i].animate.set_fill(EDGE_COLOR) for i in edge_indices],
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line 3 Highlight
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(SUCCESS_COLOR)
        )
        
        # 3. Final Meaningful visual information
        doorway_indices = [6, 7, 8, 11, 12, 13, 16, 17, 18]
        door_color = BLUE_D
        
        final_msg = Text("Meaningful Data Found!", font_size=18, color=SUCCESS_COLOR)
        # Fix for issue 29: Use place_in_area for F3 to F6
        self.place_in_area(final_msg, 'F3', 'F6', scale_factor=0.9)
        
        self.play(
            FadeOut(kernel_frame),
            FadeOut(sobel_name),
            *[squares[i].animate.set_fill(door_color) for i in doorway_indices],
            FadeIn(final_msg)
        )
        
        # Bot reaction
        self.play(
            pixel_bot.animate.scale(1.2),
            pixel_bot.animate.set_color(SUCCESS_COLOR)
        )
        self.wait(2)
