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
        lecture_lines = ["Data points become sparse.", "Distance metrics lose meaning.", "Curse of dimensionality explained."]
        self.setup_layout("Real-world Application: Data Science", lecture_lines)
        
        # Load Assets
        magnifier = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifier.svg", color=WHITE)
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg", color=WHITE)
        
        # Animation 1: Sparse points
        data_points = VGroup(*[Dot(radius=0.06, color="#00CED1") for _ in range(30)])
        self.place_in_area(data_points, 'B4', 'E6', scale_factor=0.7)
        self.place_at_grid(magnifier, 'B1', scale_factor=0.5)
        
        # Animation 2: Projection
        projection_line = Line(start=self.grid['F2'], end=self.grid['F6'], color="#FF1493")
        
        # Animation 3: Curse label
        curse_label = Text("Curse of Dimensionality", font_size=20, color="#FFD700")
        self.place_in_area(curse_label, 'A1', 'A3', scale_factor=0.6)
        
        # Animation 4: Ruler
        self.place_at_grid(ruler, 'D4', scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00CED1"))
        self.play(FadeIn(magnifier), FadeIn(data_points))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF1493"))
        self.play(Create(projection_line))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFD700"))
        self.play(FadeIn(ruler), Write(curse_label))
        self.wait(2)
