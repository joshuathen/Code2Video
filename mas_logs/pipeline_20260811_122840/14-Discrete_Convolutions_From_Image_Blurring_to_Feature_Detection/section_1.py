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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Convolution is a sliding window operator.",
            "Imagine a 3x3 filter passing over pixels.",
            "Multiply pixel values by filter weights.",
            "Sum results to produce new pixel value.",
            "This process smooths digital images."
        ]
        self.setup_layout("Intuitive Hook: The 'Moving Window' Concept", lecture_lines)
        
        # Grid visual components
        grid_pixels = VGroup(*[Square(side_length=0.7, stroke_width=2, color=WHITE) for _ in range(25)])
        grid_pixels.arrange_in_grid(rows=5, cols=5, buff=0)
        
        monitor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/monitor.svg")
        camera = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg")
        
        grid_visual_group = VGroup(grid_pixels, monitor)
        self.place_in_area(grid_visual_group, 'B2', 'F5', scale_factor=0.85)

        # 3x3 window
        window = Square(side_length=2.1, stroke_width=6, color="#FF0000")
        sliding_window_group = VGroup(window)
        self.place_in_area(sliding_window_group, 'C3', 'E5', scale_factor=0.9)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(grid_visual_group))
        self.lecture[0].set_color("#FF5733")

        # === Animation for Lecture Line 2 ===
        self.play(Create(window))
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#33FF57")

        # === Animation for Lecture Line 3 ===
        self.play(Indicate(window, color="#FF0000", scale_factor=1.05))
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#3357FF")

        # === Animation for Lecture Line 4 ===
        self.play(sliding_window_group.animate.shift(LEFT * 0.7), run_time=1)
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color("#F333FF")

        # === Animation for Lecture Line 5 ===
        self.place_at_grid(camera, 'D6', scale_factor=0.5)
        self.play(Flash(sliding_window_group.get_center(), color="#FFFFFF", line_length=0.2), FadeIn(camera))
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color("#FFD700")
        self.wait(1)
