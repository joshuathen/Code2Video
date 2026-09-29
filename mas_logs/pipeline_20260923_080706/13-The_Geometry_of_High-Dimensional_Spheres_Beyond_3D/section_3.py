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
            "Most volume concentrates near the hypersphere's surface.",
            "The interior core remains surprisingly empty.",
            "Imagine an orange with almost no fruit."
        ]
        self.setup_layout("The Paradox of Concentration of Measure", lecture_lines)
        
        # Animations
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        cube = Square(side_length=2.5, color="#FFFFFF")
        self.place_in_area(cube, "B2", "E5", scale_factor=0.9)
        self.play(Create(cube))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(GRAY)
        self.lecture[1].set_color("#FFD700")
        sphere = Circle(radius=1.25, color="#FFD700")
        self.place_in_area(sphere, "B2", "E5", scale_factor=0.9)
        self.play(Create(sphere))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(GRAY)
        self.lecture[2].set_color("#FF4500")
        # Visual representation of volume concentration (the 'crust')
        # Use SVG asset as instructed
        orange_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/orange.svg")
        orange_asset.set_color("#FF4500")
        
        crust = Annulus(inner_radius=1.0, outer_radius=1.25, color="#FF4500", fill_opacity=1)
        self.place_in_area(crust, "B2", "E5", scale_factor=1.0)
        self.place_in_area(orange_asset, "B2", "E5", scale_factor=1.0)
        
        self.play(ReplacementTransform(sphere, crust), FadeIn(orange_asset))
        self.wait(2)
