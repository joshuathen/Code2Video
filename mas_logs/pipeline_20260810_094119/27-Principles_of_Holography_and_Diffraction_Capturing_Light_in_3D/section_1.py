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
        lecture_lines = ["Light travels as an oscillating wave.", "Superposition creates constructive interference peaks.", "Phase encodes the information."]
        self.setup_layout("Prerequisite: The Wave Nature of Light", lecture_lines)
        
        # Create elements using assets
        source = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg")
        self.place_at_grid(source, 'C4', scale_factor=0.8)
        
        # Animations
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        waves = VGroup(*[Circle(radius=0.1 + i * 0.2, color="#00FFFF", stroke_width=2) for i in range(5)])
        waves.move_to(source.get_center())
        self.play(FadeIn(source))
        self.play(LaggedStart(*[Create(w) for w in waves], run_time=2))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        slits = VGroup(Line(UP*0.5, DOWN*0.5, color=YELLOW), Line(UP*0.5, DOWN*0.5, color=YELLOW))
        self.place_in_area(slits, 'B6', 'E6', scale_factor=0.5)
        self.play(FadeIn(slits))
        
        interference = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg")
        self.place_at_grid(interference, 'D3', scale_factor=0.9)
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(PINK))
        self.play(FadeIn(interference))
        self.wait(2)
