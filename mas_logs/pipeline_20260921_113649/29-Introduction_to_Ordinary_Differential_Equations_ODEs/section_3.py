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
        self.setup_layout("Visualization: The Direction Field", [
            "Solving ODEs means finding paths.",
            "Slope fields represent local rates.",
            "Solution curves trace through the field."
        ])

        # Assets
        terrain_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/terrain.svg")
        compass_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        map_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/map.svg")

        # Direction Field Setup
        field_area = VGroup()
        for x in np.linspace(-1.5, 1.5, 8):
            for y in np.linspace(-1.5, 1.5, 8):
                slope = y # dy/dx = y
                angle = np.arctan(slope)
                line = Line(start=LEFT*0.1, end=RIGHT*0.1).rotate(angle)
                line.shift(np.array([x, y, 0]))
                field_area.add(line)
        
        # Applying requested layout fix: 'B2', 'F6', scale 1.0
        self.place_in_area(field_area, 'B2', 'F6', scale_factor=1.0)

        # === Animation for Lecture Line 1 ===
        # Create a grid of slope markers using [Asset: terrain.svg]. #FFFFFF.
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.place_at_grid(terrain_icon, 'A2', scale_factor=0.3)
        self.play(Create(field_area), FadeIn(terrain_icon))

        # === Animation for Lecture Line 2 ===
        # Show vectors rotating based on f(x, y) using [Asset: compass.svg]. #00FF00.
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        self.place_at_grid(compass_icon, 'A4', scale_factor=0.3)
        self.play(field_area.animate.set_color("#00FF00"), FadeIn(compass_icon))

        # === Animation for Lecture Line 3 ===
        # Draw a path following the vectors. #FF00FF.
        # Compare with specific solution curve on a [Asset: map.svg]. #FFFFFF.
        self.play(self.lecture[2].animate.set_color("#FF00FF"))
        path = FunctionGraph(lambda x: np.exp(x), x_range=[-1.5, 1.0])
        path.set_color("#FF00FF")
        
        # Applying requested layout fix: 'B2', 'F6', scale 1.0
        self.place_in_area(path, 'B2', 'F6', scale_factor=1.0)
        self.place_at_grid(map_icon, 'A6', scale_factor=0.3)
        self.play(Create(path), FadeIn(map_icon))
        self.wait(2)
