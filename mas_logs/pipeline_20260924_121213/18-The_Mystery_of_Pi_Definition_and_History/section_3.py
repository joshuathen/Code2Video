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
        lecture_lines = ["Ancient civilizations estimated Pi values.", "Archimedes used polygons to refine it.", "Accuracy grew with more sides."]
        self.setup_layout("Historical Evolution: From Estimation to Precision", lecture_lines)
        
        # Load assets
        tablet = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/tablet.svg")
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")

        # === Animation for Lecture Line 1 ===
        # Fade in sketch using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/tablet.svg].
        self.place_at_grid(tablet, 'B5', scale_factor=0.6)
        self.play(FadeIn(tablet))
        self.lecture[0].set_color("#DEB887")

        # === Animation for Lecture Line 2 ===
        # Highlight polygon sides in white (#FFFFFF) and morph from 6 to 96.
        circle = Circle(radius=0.8, color=WHITE)
        polygon = RegularPolygon(n=6, radius=0.7, color="#D3D3D3")
        archimedes_group = VGroup(circle, polygon)
        self.place_at_grid(archimedes_group, 'D5', scale_factor=0.8)
        self.play(Create(archimedes_group))
        self.lecture[1].set_color("#D3D3D3")
        
        # Update polygon sides
        new_polygon = RegularPolygon(n=96, radius=0.7, color="#D3D3D3")
        self.play(ReplacementTransform(polygon, new_polygon))
        archimedes_group.remove(polygon)
        archimedes_group.add(new_polygon)

        # === Animation for Lecture Line 3 ===
        # Flash 'Precision' text in gold (#FFD700) with [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg].
        precision_text = Text("Precision", font_size=36, color="#FFD700")
        self.place_at_grid(precision_text, 'D2', scale_factor=0.6)
        self.place_at_grid(compass, 'D3', scale_factor=0.6)
        self.play(Flash(precision_text), FadeIn(compass))
        self.lecture[2].set_color("#FFFFFF")
        
        self.wait(2)
