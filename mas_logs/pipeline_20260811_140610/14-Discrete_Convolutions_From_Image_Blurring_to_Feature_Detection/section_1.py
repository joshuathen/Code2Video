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
        self.setup_layout("Prerequisites & Intuition", [
            "A kernel is a small, sliding window.",
            "Matrices store our input image data.",
            "Convolution performs element-wise multiplication and summation."
        ])

        # Assets
        # Grid SVG and Camera SVG
        input_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg", color=WHITE)
        camera_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg", color="#FF5733")

        # 1. Initialize empty input grid
        input_grid = VGroup(*[Square(side_length=0.7, color=WHITE) for _ in range(9)])
        input_grid.arrange_in_grid(3, 3, buff=0)
        # B019, B008, B004: Position input grid in B3-D5 area
        self.place_in_area(input_grid, "B3", "D5", scale_factor=0.9)
        input_label = Text("Input", font_size=20)
        input_label.scale(0.7) # B020
        input_label.next_to(input_grid, UP)
        
        # 2. Create 3x3 kernel
        kernel = VGroup(*[Square(side_length=0.23, color="#FF5733", fill_opacity=0.5) for _ in range(9)])
        kernel.arrange_in_grid(3, 3, buff=0)
        # Apply kernel overlay asset
        kernel_overlay = camera_icon.copy()
        kernel_overlay.scale(0.3)
        kernel_overlay.move_to(kernel.get_center())
        kernel_group = VGroup(kernel, kernel_overlay)
        self.place_at_grid(kernel_group, "D2", scale_factor=0.6) # Updated per fix
        kernel_label = Text("Kernel", font_size=20, color="#FF5733")
        kernel_label.scale(0.7) # B020
        kernel_label.next_to(kernel_group, DOWN)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF5733"))
        self.play(Create(kernel_group), Write(kernel_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFFFF"))
        self.play(FadeIn(input_grid), Write(input_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        # Simulating convolution action: moving kernel
        self.play(kernel_group.animate.shift(RIGHT * 1.5))
        self.wait(0.5)
        self.play(FadeOut(kernel_group), FadeOut(kernel_label))
        result = Text("Sum: 15", font_size=24, color="#00FF00")
        self.place_at_grid(result, "E5", scale_factor=0.8) # Updated per fix
        self.play(Write(result))
        self.wait(2)
