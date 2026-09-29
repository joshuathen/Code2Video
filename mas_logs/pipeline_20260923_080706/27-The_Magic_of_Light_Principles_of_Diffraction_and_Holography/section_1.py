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
        self.setup_layout("Prerequisite: Wave Superposition", ["Light waves superimpose when they meet.", "This creates complex interference patterns.", "Imagine ripples in a quiet pond."])
        
        # Load asset
        pond = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pond.svg")
        self.place_in_area(pond, 'A1', 'F6', scale_factor=0.5)
        self.add(pond)
        
        # Waves representation
        wave1 = FunctionGraph(lambda x: 0.2 * np.sin(x*10), x_range=[-1, 1], color=WHITE)
        wave2 = FunctionGraph(lambda x: 0.2 * np.sin(x*10), x_range=[-1, 1], color=WHITE)
        wave_group = VGroup(wave1, wave2).arrange(RIGHT, buff=0.5)
        
        # Apply positioning requested by VideoCritic
        self.place_in_area(wave_group, 'B4', 'E6', scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        self.play(Create(wave1), Create(wave2))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF00FF"))
        
        # Interference zone
        interference = VGroup(*[Circle(radius=0.2 + i*0.1, color="#FFFF00").set_stroke(width=2) for i in range(3)])
        # Apply positioning requested by VideoCritic
        self.place_in_area(interference, 'C4', 'E6', scale_factor=0.5)
        
        self.play(wave1.animate.shift(RIGHT*2), wave2.animate.shift(LEFT*2))
        self.play(FadeIn(interference))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        
        # Apply positioning requested by VideoCritic
        constructive = Dot(color="#FF0000")
        destructive = Dot(color="#0000FF")
        self.place_at_grid(constructive, 'A5', scale_factor=0.7)
        self.place_at_grid(destructive, 'F5', scale_factor=0.7)
        
        self.play(FadeIn(constructive), FadeIn(destructive))
        self.wait(2)
