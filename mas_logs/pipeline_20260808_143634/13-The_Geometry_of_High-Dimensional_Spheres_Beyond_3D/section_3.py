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
        self.setup_layout("The Corner Problem: Cubes vs. Spheres", 
                          ["Compare a hypercube to an inscribed hypersphere.", 
                           "High-dimensional volume accumulates in the cube's corners.", 
                           "The sphere occupies a tiny fraction of the cube."])
        
        # Load Assets
        cube = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cube.svg")
        sphere = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg")
        square = Square(side_length=2, color=WHITE)
        circle = Circle(radius=1, color=WHITE)
        
        # === Animation for Lecture Line 1 ===
        # Display 2D square with circle
        self.place_at_grid(square, 'C2', scale_factor=0.5)
        self.place_at_grid(circle, 'C2', scale_factor=0.5)
        self.play(FadeIn(square), FadeIn(circle))
        self.lecture[0].set_color("#00FF00")
        
        # Transition to 3D cube and sphere
        self.place_at_grid(cube, 'C5', scale_factor=0.5)
        self.place_at_grid(sphere, 'C5', scale_factor=0.5)
        sphere.set_color("#00FF00")
        self.play(FadeOut(square), FadeOut(circle), FadeIn(cube), FadeIn(sphere))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFF00")
        # Highlight corners (conceptually)
        # Using simple indicators for corners of the cube SVG
        corners = VGroup(*[Dot(color="#FFFF00", radius=0.05) for _ in range(8)])
        for i, pos in enumerate(['B4', 'B6', 'D4', 'D6', 'B4', 'B6', 'D4', 'D6']): # Simplified
            self.place_at_grid(corners[i], pos)
        self.play(Create(corners))
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF0000")
        # Show ratio shrinking
        ratio_text = Text("Ratio -> 0", font_size=24, color="#FF0000")
        self.place_at_grid(ratio_text, 'F3')
        self.play(Write(ratio_text))
        self.play(sphere.animate.scale(0.3))
        self.wait(2)
