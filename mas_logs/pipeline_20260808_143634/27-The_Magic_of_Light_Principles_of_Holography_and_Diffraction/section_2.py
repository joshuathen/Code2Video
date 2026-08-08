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
        self.setup_layout("Understanding Diffraction", [
            "Light bends when encountering obstacles.", 
            "Huygens-Fresnel explains secondary wavelets.", 
            "Apertures create characteristic diffraction patterns."
        ])

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        
        # Plane waves (using area D1-F6 as requested by Issue 24/37)
        waves = VGroup(*[Line(UP*1, DOWN*1, color=BLUE).shift(i*0.3*RIGHT) for i in range(5)])
        self.place_in_area(waves, 'D1', 'F6', scale_factor=0.6)
        self.add(waves)
        self.play(waves.animate.shift(RIGHT*1), run_time=2)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFCC00"))
        
        # Load asset /scratch/pawsey1357/jthen/Code2Video/assets/icon/slit.svg
        obstacle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/slit.svg")
        self.place_at_grid(obstacle, 'B2', scale_factor=0.5)
        self.add(obstacle)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF6666"))
        
        # Circular waves from slit
        ripples = VGroup()
        for i in range(1, 4):
            arc = Arc(radius=i*0.4, start_angle=-PI/3, angle=2*PI/3, color=BLUE)
            ripples.add(arc)
        self.place_at_grid(ripples, 'C4', scale_factor=0.6)
        self.play(Create(ripples), run_time=2)
        self.wait(2)
