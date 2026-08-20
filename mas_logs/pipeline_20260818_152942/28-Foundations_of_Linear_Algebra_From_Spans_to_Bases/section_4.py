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
        self.setup_layout("Bases: The Minimalist Framework", [
            "A basis spans the space.", 
            "Vectors must be linearly independent.", 
            "It is a minimal set.", 
            "Basis defines our coordinate system.", 
            "Basis points are always unique."
        ])

        # Assets
        asset_vectors = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/vectors.svg")
        asset_grid = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        asset_points = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/points.svg")

        # === Animation for Lecture Line 1 ===
        # Display two linearly independent vectors. Color: #FFD700
        # Incorporating asset: asset_vectors
        v1 = Vector([1, 0.5], color="#FFD700")
        v2 = Vector([-0.5, 1], color="#FFD700")
        v_group = VGroup(v1, v2, asset_vectors)
        self.place_at_grid(v_group, 'C2', scale_factor=0.7)
        self.play(Create(v1), Create(v2), FadeIn(asset_vectors))
        self.play(self.lecture[0].animate.set_color("#FFD700"))

        # === Animation for Lecture Line 2 ===
        # Grid formed by these basis vectors. Color: #00BFFF
        # Incorporating asset: asset_grid
        axes = NumberPlane(
            x_range=[-3, 3, 1], y_range=[-3, 3, 1],
            background_line_style={"stroke_color": "#00BFFF", "stroke_opacity": 0.3}
        )
        self.place_in_area(axes, 'D2', 'F6', scale_factor=0.5)
        self.place_at_grid(asset_grid, 'B3', scale_factor=0.5)
        self.play(FadeIn(axes), FadeIn(asset_grid))
        self.play(self.lecture[1].animate.set_color("#00BFFF"))

        # === Animation for Lecture Line 3 ===
        # Show how any point is reachable. Color: #FF4500
        # Incorporating asset: asset_points
        point = Dot(color="#FF4500")
        point.move_to(axes.c2p(1.5, 2))
        self.place_at_grid(asset_points, 'B4', scale_factor=0.5)
        self.play(FadeIn(point), FadeIn(asset_points))
        self.play(self.lecture[2].animate.set_color("#FF4500"))
        
        # Keep things on screen for a bit
        self.wait(2)
