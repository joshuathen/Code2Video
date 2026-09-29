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
        self.setup_layout("The Problem: Non-Normal Reality", [
            "Real-world data is often messy or skewed.",
            "Skewed data defies simple, predictable patterns.",
            "Predicting outcomes here remains a major challenge."
        ])

        # Create a bimodal distribution approximation
        def get_dist():
            # Create a histogram-like shape
            axes = Axes(x_range=[-3, 3, 1], y_range=[0, 2, 0.5], axis_config={"include_tip": False})
            axes.set_color(WHITE)
            
            # Bimodal data points
            dist = VGroup()
            for x in np.linspace(-2.5, -0.5, 20):
                height = np.exp(-(x + 1.5)**2 / 0.2)
                bar = Rectangle(height=height, width=0.1, fill_opacity=0.8, color="#FF00FF", stroke_width=0)
                bar.next_to(axes.c2p(x, 0), UP, buff=0)
                dist.add(bar)
            for x in np.linspace(0.5, 2.5, 20):
                height = np.exp(-(x - 1.5)**2 / 0.5)
                bar = Rectangle(height=height, width=0.1, fill_opacity=0.8, color="#FF00FF", stroke_width=0)
                bar.next_to(axes.c2p(x, 0), UP, buff=0)
                dist.add(bar)
            return VGroup(axes, dist)

        # === Animation for Lecture Line 1 ===
        dist_group = get_dist()
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg
        computer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg")
        comp_dist = VGroup(dist_group, computer)
        self.place_in_area(comp_dist, 'A3', 'C6', scale_factor=0.6)
        self.play(Create(comp_dist))
        self.lecture[0].set_color("#FF00FF")

        # === Animation for Lecture Line 2 ===
        # Show samples as dots
        dots = VGroup(*[Dot(color=YELLOW, radius=0.04) for _ in range(15)])
        for dot in dots:
            dot.move_to(dist_group[0].c2p(np.random.uniform(-2.5, 2.5), np.random.uniform(0.1, 1.2)))
        
        self.play(FadeIn(dots, lag_ratio=0.1))
        self.lecture[1].set_color("#FFFF00")

        # === Animation for Lecture Line 3 ===
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/scanner.svg
        scanner = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/scanner.svg")
        q_mark = Text("?", font_size=72, color=RED)
        analysis_group = VGroup(q_mark, scanner)
        self.place_at_grid(analysis_group, 'E4', scale_factor=0.75)
        self.play(Indicate(analysis_group))
        self.lecture[2].set_color(RED)
        self.wait(2)
