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
            "Complex numbers form a flexible 2D plane.",
            "Functions transform this plane like rubber sheets.",
            "Squaring inputs creates a swirling vortex."
        ]
        self.setup_layout("Prerequisites: The Geometry of Complex Numbers", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Load asset /scratch/pawsey1357/jthen/Code2Video/assets/icon/sheet.svg
        # Display complex plane (x, y) with color #FFFFFF using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/sheet.svg].
        sheet = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sheet.svg")
        self.place_at_grid(sheet, "C3", scale_factor=0.5)
        self.add(sheet)

        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            axis_config={"color": "#FFFFFF", "include_tip": True}
        )
        self.place_in_area(axes, "B2", "E5", scale_factor=0.4)
        self.add(axes)
        
        self.lecture[0].set_color("#00FFFF")
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Plot point z at (2, 1) and label it 'z'
        # Fixes for critic: Z_label at C4, Z_point at C3
        z_point = Dot(axes.c2p(2, 1), color="#FFFF00")
        self.place_at_grid(z_point, "C3", scale_factor=0.7)
        
        z_label = Text("z", color="#FFFF00", font_size=24)
        self.place_at_grid(z_label, "C4", scale_factor=0.6)
        
        self.add(z_point, z_label)
        
        self.lecture[1].set_color("#00FF00")
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Show vector from origin to z
        vector = Arrow(axes.c2p(0, 0), axes.c2p(2, 1), color="#FF00FF", buff=0)
        self.add(vector)
        
        self.lecture[2].set_color("#FF00FF")
        self.wait(2)
