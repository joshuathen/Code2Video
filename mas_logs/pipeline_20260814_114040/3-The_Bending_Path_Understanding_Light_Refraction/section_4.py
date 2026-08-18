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
        lecture_lines = ["Refraction shifts how we see.", "Objects appear deeper than they are.", "Aim lower to hit the target.", "The heron adjusts for this.", "Predators must master this effect."]
        self.setup_layout("Real-World Application: Predator Precision", lecture_lines)
        
        # Assets
        water = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/water.svg", color=BLUE)
        prey = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/insect.svg", color=ORANGE)
        
        # Composition
        content = VGroup(water, prey)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        self.place_in_area(water, 'B4', 'C6', scale_factor=0.6)
        self.play(FadeIn(water))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFD700"))
        self.place_at_grid(prey, 'B4', scale_factor=0.5)
        self.play(FadeIn(prey))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFD700"))
        jet = Line(start=water.get_center(), end=prey.get_center(), color=WHITE)
        self.play(Create(jet))
        
        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FFD700"))
        heron = Dot(color=RED)
        self.place_at_grid(heron, 'E4')
        self.play(FadeIn(heron))
        
        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FFD700"))
        self.play(heron.animate.shift(UP * 0.5), run_time=1)
