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
        lecture_lines = ["Euler's formula connects circles and sine.", "A vector rotating creates a wave.", "Speed determines the frequency observed."]
        self.setup_layout("Prerequisite Recap: Rotating Vectors", lecture_lines)
        
        # Load Compass Asset
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        self.place_at_grid(compass, 'A3', scale_factor=0.5)
        
        # Basis vectors
        i_hat = Vector(RIGHT, color="#FFFF00")
        j_hat = Vector(UP, color="#00FFFF")
        
        # Setup initial positions
        self.place_at_grid(i_hat, 'D3', scale_factor=0.7)
        self.place_at_grid(j_hat, 'D4', scale_factor=0.7)
        
        # Labels
        i_label = Tex('i-hat').next_to(i_hat.get_end(), DOWN).scale(0.7)
        j_label = Tex('j-hat').next_to(j_hat.get_end(), RIGHT).scale(0.7)
        self.add(i_label, j_label)
        
        # Group them for rotation
        vectors = VGroup(i_hat, j_hat)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        self.play(FadeIn(compass), Create(i_hat), Create(j_hat), Write(i_label), Write(j_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        
        # Path
        arc_i = Arc(radius=0.7, start_angle=0, angle=PI/2, color="#AAAAAA")
        arc_i.move_to(self.grid['D3'])
        
        arc_j = Arc(radius=0.7, start_angle=PI/2, angle=PI/2, color="#AAAAAA")
        arc_j.move_to(self.grid['D4'])
        
        self.play(Create(arc_i), Create(arc_j))
        
        # Rotate
        self.play(Rotate(i_hat, angle=PI/2, about_point=self.grid['D3']),
                  Rotate(j_hat, angle=PI/2, about_point=self.grid['D4']),
                  FadeOut(i_label), FadeOut(j_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#AAAAAA"))
        self.wait(2)
