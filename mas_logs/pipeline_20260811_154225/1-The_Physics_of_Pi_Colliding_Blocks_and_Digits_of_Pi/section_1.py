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
            "Two blocks collide; Pi emerges in the count.",
            "How do mechanical collisions encode circle geometry?",
            "We track collisions as digits of Pi.",
            "Mass ratios dictate the number of bounces.",
            "Discover the surprising link between math and physics."
        ]
        self.setup_layout("The Hook: The Pi Mystery", lecture_lines)
        
        # Assets
        track = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/track.svg", color="#FFFFFF")
        blocks = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/blocks.svg", color="#FFD700")
        label = Text("Particle Collider", font_size=20, color="#FFD700")
        point = Dot(color="#FF6347")
        reflection_vec = Arrow(start=ORIGIN, end=RIGHT, color="#00CED1")
        bounce_mark = Dot(color="#FF4500")

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.place_in_area(track, 'B3', 'D5', scale_factor=0.5)
        self.place_at_grid(blocks, 'B3', scale_factor=0.7)
        self.play(Create(track), Write(blocks))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF6347"))
        self.place_at_grid(point, 'A4', scale_factor=0.7)
        self.play(FadeIn(point))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00CED1"))
        self.place_at_grid(reflection_vec, 'C4', scale_factor=0.7)
        self.play(GrowArrow(reflection_vec))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FF4500"))
        self.place_at_grid(bounce_mark, 'C4', scale_factor=0.8)
        self.play(FadeIn(bounce_mark))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FFFF00"))
        self.wait(1)
