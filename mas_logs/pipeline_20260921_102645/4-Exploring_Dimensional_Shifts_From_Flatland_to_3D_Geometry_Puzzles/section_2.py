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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Geometry of Flattening: Cross-Sections", [
            "Slicing a 3D object reveals interior structure.",
            "A sphere becomes a circle when sliced.",
            "Slices change as they pass through.",
            "We can infer volume from 2D slices.",
            "Interior geometry is hidden in plain sight."
        ])

        # Assets
        sphere_icon = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg"
        
        # Create objects
        sphere = SVGMobject(sphere_icon).set_color(WHITE)
        plane = Square(side_length=2, fill_opacity=0.5, color="#FF00FF")
        plane.rotate(PI/2, axis=RIGHT)
        
        # Cross section circle
        circle = Circle(radius=0.5, color="#00FF00", fill_opacity=0.8)
        
        visual_group = VGroup(sphere, plane)
        self.place_in_area(visual_group, 'B3', 'E6', scale_factor=0.9)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        self.play(FadeIn(sphere))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        self.play(FadeIn(plane))
        self.place_at_grid(circle, 'D5', scale_factor=0.6)
        self.play(FadeIn(circle))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF00FF"))
        # Slicing movement
        plane_tracker = ValueTracker(1)
        
        self.play(
            plane.animate.shift(UP * 2),
            run_time=3
        )
        
        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#00FFFF"))
        self.wait(1)
        
        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FF6600"))
        self.wait(2)
