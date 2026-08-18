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
        self.setup_layout("Application: Fractals and Data Mapping", ["Curves map 2D to 1D.", "Preserving spatial data locality.", "Optimizing computer memory storage."])
        
        # Assets
        computer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg", color=WHITE)
        memory = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/memory.svg", color=WHITE)
        
        # 1. Map complex data to 1D sequence
        square = Square(side_length=2, color=BLUE).set_fill(BLUE, opacity=0.3)
        self.place_in_area(square, 'B1', 'C3', scale_factor=0.5)
        
        line = Line(start=LEFT, end=RIGHT, color=YELLOW)
        self.place_in_area(line, 'E1', 'E3', scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        self.place_at_grid(computer, 'B5', scale_factor=0.6)
        self.play(FadeIn(square), FadeIn(line), FadeIn(computer))
        self.lecture[0].set_color(BLUE)

        # 2. Illustrate spatial locality of data
        dot_2d = Dot(color=RED)
        dot_1d = Dot(color=RED)
        dot_2d.move_to(square.get_center())
        dot_1d.move_to(line.get_left())
        
        # === Animation for Lecture Line 2 ===
        self.play(Create(dot_2d), Create(dot_1d))
        self.play(dot_1d.animate.move_to(line.get_right()), run_time=2)
        self.lecture[1].set_color(YELLOW)

        # 3. Show cluster preservation in 1D
        self.place_at_grid(memory, 'E5', scale_factor=0.6)
        cluster = VGroup(*[Dot(point=square.get_center() + np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), 0]), color=GREEN) for _ in range(5)])
        
        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(cluster), FadeIn(memory))
        self.lecture[2].set_color(GREEN)
        self.wait(2)
