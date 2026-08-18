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
        self.setup_layout("The Physics of Diffraction: Light Bending", 
                          ["Diffraction occurs when light hits obstacles.", 
                           "Huygens principle defines wavefronts as sources.", 
                           "Light spreads into patterns of fringes."])
        self.lecture.set_opacity(1)

        # Animation Setup
        # Wavefronts
        waves = VGroup(*[Line(UP*0.5, DOWN*0.5, color=WHITE).shift(LEFT*i*0.3) for i in range(5)])
        self.place_in_area(waves, 'D2', 'F5', scale_factor=0.5)

        # Obstacle using Asset
        edge_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/edge.svg")
        edge_asset.set_color("#00FFFF")
        self.place_at_grid(edge_asset, 'C3', scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(WHITE), Write(waves))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        # Huygens point sources visualization
        points = VGroup(*[Dot(color="#00FFFF") for _ in range(5)])
        self.place_at_grid(points, "C4")
        self.play(FadeIn(points))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        # Spreading waves
        arcs = VGroup(*[Arc(radius=0.5 + i*0.2, angle=PI/2, start_angle=-PI/4, color="#00FFFF") for i in range(4)])
        self.place_at_grid(arcs, 'C4', scale_factor=0.7)
        self.play(Create(arcs))
        self.wait(2)
