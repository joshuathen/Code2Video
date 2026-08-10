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
        self.setup_layout("Visualizing Filters: Edge Detection", [
            "Different kernels produce different results.",
            "Sobel filters detect sharp contrast.",
            "See blurred images become sharp."
        ])
        
        # Create objects
        # [Asset: kernel_variety] - A square/dot for visualization
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg]
        kernel_vis = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg", color=WHITE)
        # [Asset: sobel_edge] - An outline/edge-map representation
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/photograph.svg]
        edge_map = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/photograph.svg", color=WHITE)
        
        kernel_label = Text("Input", font_size=20)
        sobel_label = Text("Edge Map", font_size=20)

        # === Animation for Lecture Line 1 ===
        # Display an image of a camera.
        self.place_in_area(kernel_vis, 'B2', 'C3', scale_factor=0.8)
        self.play(FadeIn(kernel_vis))
        self.lecture[0].set_color("#00FFFF")

        # === Animation for Lecture Line 2 ===
        # Apply a vertical filter to highlight edges.
        self.lecture[1].set_color("#FFFF00")
        kernel_vis.set_color("#FFFF00")
        self.play(kernel_vis.animate.rotate(PI/4))

        # === Animation for Lecture Line 3 ===
        # Show the final edge-map output in #FF8000.
        self.lecture[2].set_color("#FF8000")
        edge_map.set_color("#FF8000")
        self.place_in_area(edge_map, 'B4', 'C5', scale_factor=0.8)
        self.place_at_grid(kernel_label, 'D2', scale_factor=0.5)
        self.place_at_grid(sobel_label, 'D4', scale_factor=0.5)
        self.play(FadeIn(edge_map), Write(kernel_label), Write(sobel_label))
        self.wait(2)
