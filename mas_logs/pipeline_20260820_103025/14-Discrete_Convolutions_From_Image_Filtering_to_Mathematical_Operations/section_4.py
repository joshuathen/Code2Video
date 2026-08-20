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
        lecture_lines = [
            "Kernels act as blurs or sharpeners in images.",
            "CNNs learn these kernels automatically from data.",
            "They detect patterns like whiskers or ear triangles."
        ]
        self.setup_layout("Real-World Applications", lecture_lines)
        
        # Assets
        cat_img = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cat.png")
        camera_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg")
        
        # Elements
        self.place_in_area(cat_img, 'A2', 'C5', scale_factor=0.6)
        
        kernel_rect = Square(side_length=0.8, color="#00FFFF").set_fill("#00FFFF", opacity=0.3)
        self.place_at_grid(kernel_rect, 'D4', scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(cat_img))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(kernel_rect))
        self.lecture[1].set_color("#00FFFF")
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # Use camera icon instead of sliding box
        self.place_at_grid(camera_svg, 'E5', scale_factor=0.6)
        camera_svg.set_color("#FFCC00")
        
        self.play(
            kernel_rect.animate.move_to(self.grid['E5']),
            FadeIn(camera_svg),
            run_time=1.5
        )
        self.lecture[2].set_color("#FFCC00")
        self.wait(1)
