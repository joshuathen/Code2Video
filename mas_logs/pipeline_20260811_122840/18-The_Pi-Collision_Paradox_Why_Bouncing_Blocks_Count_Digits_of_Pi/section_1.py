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
            "Tiny block collides with a massive block.",
            "The number of bounces is surprisingly huge.",
            "Mass ratios unlock the digits of Pi."
        ]
        self.setup_layout("The Hook: A Paradoxical Counting Puzzle", lecture_lines)
        
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg
        wall = Line(UP, DOWN, color=GREY).scale(1.5)
        self.place_at_grid(wall, "D3", scale_factor=0.6)
        
        tiny = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg", color=WHITE, fill_opacity=1)
        huge = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg", color=WHITE, fill_opacity=1)
        
        self.place_at_grid(tiny, "D4", scale_factor=0.5)
        self.place_at_grid(huge, "D6", scale_factor=0.8)
        
        label_tiny = Text("M1", font_size=20).next_to(tiny, UP, buff=0.1)
        label_huge = Text("M2", font_size=20).next_to(huge, UP, buff=0.1)
        
        # Velocity vectors
        v_tiny = Arrow(start=ORIGIN, end=RIGHT*0.5, color=WHITE)
        v_huge = Arrow(start=ORIGIN, end=LEFT*0.3, color=WHITE)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(wall), FadeIn(tiny), FadeIn(huge), FadeIn(label_tiny), FadeIn(label_huge))
        self.lecture[0].set_color("#FFFFFF")
        self.play(Create(v_tiny), Create(v_huge))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF00FF")
        # Animate movement
        self.play(
            tiny.animate.next_to(huge, LEFT, buff=0.1),
            v_tiny.animate.next_to(tiny, UP, buff=0.1),
            v_huge.animate.next_to(huge, UP, buff=0.1),
            run_time=2
        )
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00")
        # Change M2 color to #FFFF00 for emphasis as it hits M1 using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg].
        self.play(
            huge.animate.set_color("#FFFF00"),
            run_time=0.5
        )
        self.play(
            v_tiny.animate.set_color("#FFFF00"),
            v_huge.animate.set_color("#FFFF00"),
            run_time=0.5
        )
        self.play(
            v_tiny.animate.set_color(WHITE),
            v_huge.animate.set_color(WHITE),
            run_time=0.5
        )
