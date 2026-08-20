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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Different kernels produce unique effects on signals.",
            "A blurring kernel averages neighboring pixel values.",
            "An edge detection kernel highlights sharp transitions.",
            "This extracts specific patterns from the data.",
            "Convolutions act as flexible filters for processing."
        ]
        self.setup_layout("The Power of Kernels", lecture_lines)
        
        # Helpers
        grid_m = VGroup(*[Square(side_length=0.7, stroke_width=1, stroke_color=GRAY) for _ in range(16)])
        grid_m.arrange_in_grid(4, 4, buff=0)
        self.place_in_area(grid_m, "B2", "E5")
        self.add(grid_m)

        # === Animation for Lecture Line 1 ===
        # Display a grid and apply a blurring kernel in #FF00FF using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/lens.svg].
        lens = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/lens.svg", color="#FF00FF")
        self.place_at_grid(lens, "A3", scale_factor=0.5)
        self.play(self.lecture[0].animate.set_color("#FF00FF"), FadeIn(lens))

        # === Animation for Lecture Line 2 ===
        # Display a grid and apply an edge detection kernel in #00FF00.
        edge_label = Text("Blurring Kernel", color="#FF00FF", font_size=18)
        edge_label.next_to(lens, RIGHT, buff=0.1)
        self.add(edge_label)
        self.play(self.lecture[1].animate.set_color("#FF00FF"))

        # === Animation for Lecture Line 3 ===
        # Animate edge highlights appearing around a shape in the grid.
        highlight = Square(side_length=0.7, stroke_color="#00FF00", stroke_width=4)
        self.place_in_area(highlight, "C3", "C3")
        self.play(self.lecture[2].animate.set_color("#00FF00"), Create(highlight))

        # === Animation for Lecture Line 4 ===
        # Show patterns like lines and curves being isolated using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/paintbrush.svg].
        brush = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/paintbrush.svg")
        self.place_at_grid(brush, "A5", scale_factor=0.5)
        self.play(self.lecture[3].animate.set_color("#FFFF00"), FadeIn(brush))

        # === Animation for Lecture Line 5 ===
        # Represent convolutional filters as a versatile bank of tools using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/filter.svg].
        filter_tool = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/filter.svg")
        self.place_at_grid(filter_tool, "F5", scale_factor=0.5)
        self.play(self.lecture[4].animate.set_color("#00FFFF"), FadeIn(filter_tool))
        self.wait(2)
