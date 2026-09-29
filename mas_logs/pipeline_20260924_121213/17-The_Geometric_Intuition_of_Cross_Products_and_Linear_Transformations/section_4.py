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
        self.setup_layout("Geometric Application: The Normal Vector", [
            "Surface normals are vital for 3D lighting.",
            "Cross products compute these normals efficiently.",
            "Light reflection depends on surface orientation.",
            "Lighting calculations use normals for pixel color.",
            "Physics simulations rely on these geometric tools."
        ])

        # Assets
        triangle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/triangle.svg")
        light_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/light.svg")

        # Objects
        v1 = Line3D(start=ORIGIN, end=np.array([1, 0, 0.5]), color="#FFFF00")
        v2 = Line3D(start=ORIGIN, end=np.array([0, 1, 0.5]), color="#FFFF00")
        normal = Line3D(start=ORIGIN, end=np.cross([1, 0, 0.5], [0, 1, 0.5]), color="#FF00FF")
        normal_label = MathTex(r"\vec{n}", color="#FF00FF")
        
        group = VGroup(triangle, v1, v2, normal, light_icon)
        
        # Repositioning group per VideoCritic [Issue 32, 39]
        self.place_in_area(group, 'B4', 'C6', scale_factor=0.9)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"), Write(triangle), Write(v1), Write(v2))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF00FF"), Write(normal))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF00FF"))

        # === Animation for Lecture Line 4 ===
        # Position label per VideoCritic [Issue 31, 39]
        self.place_at_grid(normal_label, 'C5', scale_factor=1.0)
        self.play(self.lecture[3].animate.set_color("#FF00FF"), Write(normal_label))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#00FFFF"), Write(light_icon))
