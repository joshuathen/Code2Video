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
        lines = [
            "Sequential processing limits memory for long sentences.",
            "RNNs forget early information as they progress.",
            "A conveyor belt model loses context over time.",
            "Attention provides a new way to process input.",
            "Parallel processing enables global context for every word."
        ]
        self.setup_layout("The Problem: Why Old Models Forget", lines)
        
        # === Animation for Lecture Line 1 ===
        problem_text = Text("Sequential Processing", font_size=36, color=WHITE)
        self.place_in_area(problem_text, 'B2', 'B5', scale_factor=1.0)
        self.play(FadeIn(problem_text))
        self.play(problem_text.animate.set_color("#FFD700"))
        self.play(self.lecture[0].animate.set_color("#FFD700"))

        # === Animation for Lecture Line 2 ===
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/conveyor.svg]
        conveyor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/conveyor.svg")
        blocks = VGroup(*[Square(side_length=0.3, color=WHITE, fill_opacity=0.6) for _ in range(5)])
        blocks.arrange(RIGHT, buff=0.1)
        
        container = VGroup(conveyor, blocks).arrange(DOWN)
        self.place_at_grid(container, 'D2', scale_factor=0.9)
        
        self.play(FadeIn(container))
        self.play(container.animate.set_color("#4169E1"))
        self.play(self.lecture[1].animate.set_color("#4169E1"))

        # === Animation for Lecture Line 3 ===
        bottleneck = Polygon(np.array([-0.5, 0.5, 0]), np.array([0.5, 0.5, 0]), np.array([0.2, -0.5, 0]), np.array([-0.2, -0.5, 0]), color=WHITE)
        self.place_at_grid(bottleneck, 'E4', scale_factor=0.9)
        self.play(FadeIn(bottleneck))
        self.play(bottleneck.animate.set_color("#FF4500"))
        self.play(self.lecture[2].animate.set_color("#FF4500"))
        
        # === Animation for Lecture Line 4 ===
        attention_text = Text("Attention", font_size=30, color=WHITE)
        self.place_at_grid(attention_text, 'C4', scale_factor=0.8)
        self.play(FadeIn(attention_text))
        self.play(attention_text.animate.set_color("#32CD32"))
        self.play(self.lecture[3].animate.set_color("#32CD32"))

        # === Animation for Lecture Line 5 ===
        # Parallel processing using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/conveyor.svg]
        parallel_conveyor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/conveyor.svg")
        self.place_at_grid(parallel_conveyor, 'F2', scale_factor=0.7)
        self.play(FadeIn(parallel_conveyor))
        self.play(parallel_conveyor.animate.set_color("#FFFFFF"))
        self.play(self.lecture[4].animate.set_color("#FFFFFF"))
