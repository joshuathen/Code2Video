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
        self.setup_layout("Visualizing the Bending Rule", ["Denser medium, light bends inward.", "Rarer medium, light bends outward.", "The normal line is our reference."])
        
        # Assets
        laser = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg", color=YELLOW)
        prism = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/prism.svg", color=BLUE_B)
        water = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/water.svg", color=GOLD)

        # Scene elements
        normal_line = Line(start=self.grid['A4'], end=self.grid['F4'], color=GRAY)
        interface = Line(start=self.grid['C1'], end=self.grid['C6'], color=BLUE_B)
        
        i_ray = Line(start=self.grid['A2'], end=self.grid['C4'], color=YELLOW)
        r_ray_denser = Line(start=self.grid['C4'], end=self.grid['E5'], color=YELLOW)
        
        snells_law = MathTex(r"n_1 \sin(i) = n_2 \sin(r)", font_size=32, color=GOLD)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.title))
        self.place_at_grid(laser, 'A2', scale_factor=0.3)
        self.play(FadeIn(laser))
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.add(normal_line, interface)
        self.play(Create(i_ray), Create(r_ray_denser))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(YELLOW))
        self.place_at_grid(prism, 'B5', scale_factor=0.3)
        self.play(FadeIn(prism))
        r_ray_rarer = Line(start=self.grid['C4'], end=self.grid['E3'], color=YELLOW)
        self.play(Transform(r_ray_denser, r_ray_rarer))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(YELLOW))
        self.place_at_grid(water, 'E2', scale_factor=0.3)
        self.play(FadeIn(water))
        self.place_in_area(snells_law, 'B2', 'C5', scale_factor=1.0)
        self.play(Write(snells_law))
        self.wait(2)
