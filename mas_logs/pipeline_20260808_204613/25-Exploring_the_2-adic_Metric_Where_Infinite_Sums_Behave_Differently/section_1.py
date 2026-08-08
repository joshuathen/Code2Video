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
        self.setup_layout("Prerequisite: The Intuition of Convergence", 
                          ["Standard convergence means getting closer.", 
                           "Distances shrink on a number line.", 
                           "An ant approaches the crumb."])
        
        # Assets
        ant = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ant.svg")
        crumb = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/crumb.svg")
        
        # Define elements
        target = self.place_at_grid(crumb, 'C5', scale_factor=0.3)
        target_label = Text("Crumb", font_size=20, color=RED).next_to(target, DOWN)
        
        ant_mobject = self.place_at_grid(ant, 'C2', scale_factor=0.3)
        
        seq_label = Text("Sequence", font_size=20, color="#FFCC00")
        self.place_at_grid(seq_label, 'B3', scale_factor=0.7)
        seq_label.set_opacity(0)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.play(FadeIn(target), Write(target_label), FadeIn(ant_mobject))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(BLUE)
        self.play(FadeIn(seq_label))
        self.play(ant_mobject.animate.move_to(self.grid['C4'] + LEFT * 0.3))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(BLUE)
        self.play(ant_mobject.animate.move_to(target.get_center() + LEFT * 0.1))
        self.wait(1)
