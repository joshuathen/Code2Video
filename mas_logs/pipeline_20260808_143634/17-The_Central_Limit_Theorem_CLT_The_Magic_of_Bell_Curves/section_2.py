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
        self.setup_layout("The Problem: The Chaos of Small Samples", [
            "Real-world data is rarely normal.", 
            "Most distributions are skewed or erratic.", 
            "Small samples lead to inconsistent predictions."
        ])
        
        # Asset
        camera_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        text1 = Text("Small samples capture noisy snapshots.", font_size=28, color="#FFFFFF")
        self.place_in_area(text1, "A2", "B4", scale_factor=0.6)
        self.play(Write(text1))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FFFF")
        clusters = VGroup()
        for i in range(3):
            cluster = VGroup(*[Dot(radius=0.05, color="#00FFFF") for _ in range(10)])
            cluster.arrange_in_grid(2, 5, buff=0.1)
            clusters.add(cluster)
        
        self.place_at_grid(clusters[0], "B3", scale_factor=0.7)
        self.place_at_grid(clusters[1], "B5", scale_factor=0.7)
        self.place_at_grid(clusters[2], "D5", scale_factor=0.8)
        
        # Add camera icon near clusters
        camera_icon.scale(0.5).next_to(clusters[0], UP)
        
        self.play(Create(clusters), FadeIn(camera_icon))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF0000")
        chaos = Text("Chaos", font_size=40, color="#FF0000")
        self.place_at_grid(chaos, "D3", scale_factor=0.6)
        self.play(FadeIn(chaos, scale=1.5))
        self.wait(2)
