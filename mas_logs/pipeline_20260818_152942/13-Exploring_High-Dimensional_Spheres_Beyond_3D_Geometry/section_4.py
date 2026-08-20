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
        self.setup_layout(
            "Application: Data Science and Machine Learning", 
            ["High-dimensional spheres inform modern machine learning.", 
             "They define similarity through cosine metrics.", 
             "AI clustering identifies patterns in vast spaces."]
        )
        
        # Assets
        icon_computer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg")
        icon_sensor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sensor.svg")
        
        # === Animation for Lecture Line 1 ===
        points = VGroup(*[Dot(radius=0.05, color=WHITE) for _ in range(30)])
        for p in points:
            p.move_to(np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1), 0]))
        self.place_in_area(points, 'B3', 'E5', scale_factor=0.6)
        self.place_at_grid(icon_computer, 'A6', scale_factor=0.3)
        self.play(FadeIn(points), FadeIn(icon_computer))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))

        # === Animation for Lecture Line 2 ===
        pca_lines = VGroup(*[Line(start=ORIGIN, end=np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), 0]), color="#00CED1") for _ in range(5)])
        self.place_in_area(pca_lines, 'B3', 'E5', scale_factor=0.6)
        self.play(Create(pca_lines), run_time=2)
        self.play(self.lecture[1].animate.set_color("#00CED1"))

        # === Animation for Lecture Line 3 ===
        cluster = Circle(radius=0.8, color="#FFD700", fill_opacity=0.2)
        self.place_at_grid(cluster, 'D4', scale_factor=0.5)
        self.place_at_grid(icon_sensor, 'F6', scale_factor=0.3)
        self.play(DrawBorderThenFill(cluster), FadeIn(icon_sensor))
        self.play(self.lecture[2].animate.set_color("#FFD700"))
        self.wait(2)
