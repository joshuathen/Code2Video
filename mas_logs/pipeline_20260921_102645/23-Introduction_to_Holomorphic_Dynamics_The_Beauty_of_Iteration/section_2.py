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
        self.setup_layout("Fixed Points and Attractors", [
            "Fixed points are where the function remains unchanged.",
            "Orbits track a point's movement under repeated iteration.",
            "Stable fixed points attract nearby points like magnets."
        ])
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFF00")
        z0 = Dot(point=self.grid["C3"], color=WHITE)
        z0_label = Text("z_0", font_size=18)
        self.place_at_grid(z0_label, 'B3', scale_factor=0.6)
        self.add(z0, z0_label)
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFFF00")
        
        # Magnet icon
        magnet = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnet.svg")
        self.place_at_grid(magnet, 'C5', scale_factor=0.5)
        self.play(FadeIn(magnet))
        
        dots = VGroup()
        current_pos = self.grid["C3"]
        for i in range(5):
            dot = Dot(point=current_pos, color=BLUE_B).scale(0.8 - i * 0.1)
            dots.add(dot)
            current_pos = current_pos + np.array([-0.3, 0.1, 0]) 
            self.play(FadeIn(dot))
        
        orbit_group = VGroup(magnet, dots)
        self.place_in_area(orbit_group, 'A4', 'C6', scale_factor=0.9)
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#00FF00")
        
        attractor = Dot(point=self.grid["C5"], color="#00FF00")
        attractor_label = Text("Attractor", font_size=18, color="#00FF00")
        self.place_at_grid(attractor_label, 'D4', scale_factor=0.7)
        
        self.play(FadeIn(attractor), Write(attractor_label))
        self.wait(3)
