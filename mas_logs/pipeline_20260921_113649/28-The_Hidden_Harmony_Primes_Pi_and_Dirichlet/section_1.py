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
        lecture_lines = [
            "Primes seem chaotic, scattered across integers.",
            "Yet they follow a hidden, rhythmic heartbeat.",
            "The Prime Number Theorem reveals this order.",
            "Integers hide a profound structural density.",
            "Math transforms noise into predictable patterns."
        ]
        self.setup_layout("Introduction: The Chaos and Order of Primes", lecture_lines)
        
        # Assets
        cloud = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cloud.svg")
        heart = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/heartbeat.svg")
        grid = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        clouds = VGroup(*[cloud.copy().scale(0.2) for _ in range(5)])
        self.place_in_area(clouds, "A1", "F6")
        self.play(FadeIn(clouds))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF00FF")
        self.place_at_grid(heart, "C3", scale_factor=0.5)
        heart.set_color("#FF00FF")
        self.play(FadeIn(heart), heart.animate.scale(1.2).shift(UP*0.1), run_time=1)
        self.play(heart.animate.scale(1/1.2).shift(DOWN*0.1), run_time=1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FFFF")
        graph = Axes(x_range=[0, 6, 1], y_range=[0, 4, 1], axis_config={"include_tip": False})
        pnt = graph.plot(lambda x: np.log(x+0.1), color="#00FFFF")
        self.place_in_area(graph, "A1", "F6", scale_factor=0.5)
        self.play(Create(pnt))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFFF00")
        self.place_at_grid(grid, "D4", scale_factor=0.8)
        grid.set_color("#FFFF00")
        self.play(FadeIn(grid))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#00FF00")
        final_grid = grid.copy()
        final_grid.set_color("#00FF00")
        self.play(Transform(grid, final_grid))
        self.wait(1)
