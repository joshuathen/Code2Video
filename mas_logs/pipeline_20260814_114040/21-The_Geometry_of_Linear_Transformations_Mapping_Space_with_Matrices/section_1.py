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
            "Imagine coordinate grids as flexible, interactive rubber sheets.",
            "Linear transformations stretch or rotate this grid space.",
            "Vectors are simply arrows pointing within this space."
        ]
        self.setup_layout("Intuitive Hook: The 'Digital Puppet'", lecture_lines)
        
        # Load asset
        puppet = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/puppet.svg")
        puppet_label = Text("Wireframe", font_size=18, color="#00FF00")
        
        # Rotation indicator
        rot_arrow = Arc(radius=0.5, start_angle=0, angle=PI/2, color="#FF00FF")
        rot_label = Text("Rotation", font_size=18, color="#FF00FF")
        
        # Vector
        vector = Arrow(start=ORIGIN, end=RIGHT*1.5 + UP*1, color="#00FFFF")
        vec_label = Text("Coordinates", font_size=18, color="#00FFFF")

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        # Using puppet asset, fixing positioning per critique
        puppet.set_color("#00FF00")
        self.place_in_area(puppet, 'B1', 'B3', scale_factor=0.6)
        self.place_at_grid(puppet_label, 'A4')
        self.play(FadeIn(puppet), Write(puppet_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF00FF"))
        # Using rotation arrow and puppet
        self.place_at_grid(rot_arrow, 'C2', scale_factor=0.7)
        self.place_at_grid(rot_label, 'B5')
        self.play(Create(rot_arrow), Write(rot_label))
        self.play(Rotate(puppet, angle=PI/6, about_point=self.grid['C2']))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        # Using vector
        self.place_in_area(vector, 'D3', 'E5', scale_factor=0.6)
        self.place_at_grid(vec_label, 'F5')
        self.play(GrowArrow(vector), Write(vec_label))
        self.wait(2)
