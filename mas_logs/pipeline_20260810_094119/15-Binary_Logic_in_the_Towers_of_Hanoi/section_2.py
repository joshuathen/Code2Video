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
        lecture_lines = ["The puzzle has three rods and stacked discs.", 
                         "Move one disc at a time between rods.", 
                         "Never place a larger disc on a smaller one.", 
                         "The goal is moving all discs to another rod."]
        self.setup_layout("The Towers of Hanoi Puzzle", lecture_lines)
        
        # Setup Puzzle Assets
        # Using SVGMobject for disk and rod as specified in assets
        rod_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/rod.svg"
        disc_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/disc.svg"
        
        rods = VGroup(*[SVGMobject(rod_path) for _ in range(3)])
        # Fixing layout: moved to 'D4'
        self.place_at_grid(rods, 'D4', scale_factor=0.6)
        
        # Rod Labels
        labels = VGroup(*[Text(label, font_size=24) for label in ["A", "B", "C"]])
        for i, l in enumerate(labels):
            l.next_to(rods[i], DOWN, buff=0.1)
            self.add(l)
        
        # 3 Discs
        discs = VGroup(*[SVGMobject(disc_path, color=color) 
                         for color in [RED, GREEN, BLUE]])
        discs.arrange(UP, buff=0)
        
        hanoi_group = VGroup(rods, discs)
        # Using place_in_area as requested
        self.place_in_area(hanoi_group, 'C4', 'E6', scale_factor=0.65)
        
        self.play(FadeIn(rods), FadeIn(discs))
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        
        # Highlight largest disk
        highlight = SurroundingRectangle(discs[0], color=YELLOW)
        self.play(Create(highlight))
        self.wait(1)
        self.play(FadeOut(highlight))
        
        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(YELLOW))
        
        # Animate move of largest disk to tower C (simplified)
        self.play(discs[0].animate.move_to(rods[2].get_center() + UP*0.5))
        self.wait(2)
