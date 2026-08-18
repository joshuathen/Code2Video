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

class Section4Scene(ThreeDScene, TeachingScene):
    def construct(self):
        self.setup_layout("The Visual Transformation", [
            "Tilt the plane to change the shape.",
            "The circle transforms into an ellipse.",
            "Steeper cuts create parabolas and hyperbolas.",
            "Spheres shift to match the new slice.",
            "Every cut maintains its fundamental property."
        ])

        # Define objects - replacing with SVG assets as requested
        cone = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cone.svg")
        plane = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/plane.svg")
        
        # Position objects
        self.place_at_grid(cone, 'D3', scale_factor=0.7)
        self.place_at_grid(plane, 'D3', scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        self.play(FadeIn(cone), FadeIn(plane))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF00FF")
        # Animate tilt
        self.play(Rotate(plane, angle=PI/6, about_point=self.grid['D3'], axis=UP))
        ellipse_label = Text("Ellipse", font_size=20, color="#FF00FF")
        self.place_at_grid(ellipse_label, 'E3', scale_factor=0.8) # Fix: issue 30/45
        self.play(Write(ellipse_label))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FFFF")
        self.play(Rotate(plane, angle=PI/6, about_point=self.grid['D3'], axis=UP))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#00FF00")
        sphere = Sphere(radius=0.3, color="#00FF00").set_opacity(0.6)
        self.place_at_grid(sphere, 'D4', scale_factor=0.6) # Fix: issue 31/46
        self.play(FadeIn(sphere))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFFF00")
        focus_dot = Dot(color="#FFFF00")
        self.place_at_grid(focus_dot, 'D2', scale_factor=0.5) # Fix: issue 32/47
        self.play(FadeIn(focus_dot))
        self.wait(1)
