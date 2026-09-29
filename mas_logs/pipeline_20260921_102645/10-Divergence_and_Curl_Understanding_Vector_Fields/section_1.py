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
        self.setup_layout("Vector Fields: Intuition", [
            "Vector fields assign vectors to points in space.",
            "Imagine velocity at every point in a river.",
            "Tiny cross probes visualize the flowing motion."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Vector fields assign vectors to points in space.
        self.play(self.lecture[0].animate.set_color("#00ffff"))
        
        # Draw vector field grid with color #aaaaaa using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/river.svg].
        river = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/river.svg", color="#aaaaaa")
        self.place_in_area(river, 'B3', 'F6', scale_factor=0.5)
        self.play(FadeIn(river))

        # === Animation for Lecture Line 2 ===
        # Imagine velocity at every point in a river.
        self.play(self.lecture[1].animate.set_color("#ff00ff"))
        
        # No additional asset required by storyboard here, just color change
        self.play(river.animate.set_color("#ff00ff"))

        # === Animation for Lecture Line 3 ===
        # Tiny cross probes visualize the flowing motion.
        self.play(self.lecture[2].animate.set_color("#ffff00"))
        
        # Animate path tracing through field #ff00ff using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/probe.svg].
        probe = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/probe.svg", color="#ffff00")
        self.place_at_grid(probe, 'D4', scale_factor=0.7)
        self.play(Create(probe))
        self.play(probe.animate.shift(RIGHT*1 + UP*0.5), run_time=2)
