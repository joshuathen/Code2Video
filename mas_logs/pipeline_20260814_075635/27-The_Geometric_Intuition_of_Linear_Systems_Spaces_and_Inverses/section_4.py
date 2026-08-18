from manim import *
import os

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
        lecture_lines = ["Inverse matrices reverse the transformation.", "They act as a rewind button.", "Only non-collapsing transformations are invertible."]
        self.setup_layout("Inverse Matrices: The 'Rewind' Button", lecture_lines)
        
        dot = Dot(color=YELLOW)
        self.place_at_grid(dot, 'C2', scale_factor=0.6)
        
        vector = Arrow(start=ORIGIN, end=dot.get_center(), buff=0, color=BLUE)
        self.add(vector)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.play(vector.animate.put_start_and_end_on(ORIGIN, self.grid['C5']), dot.animate.move_to(self.grid['C5']))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        # Use remote icon as rewind button
        remote_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/remote.svg")
        self.place_at_grid(remote_icon, 'E3', scale_factor=0.5)
        self.play(FadeIn(remote_icon))
        self.play(vector.animate.put_start_and_end_on(ORIGIN, self.grid['C2']), dot.animate.move_to(self.grid['C2']), run_time=1.5)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(RED))
        rect = Rectangle(width=2.5, height=2.5, color=WHITE).set_fill(GREY, opacity=0.3)
        self.place_in_area(rect, 'B3', 'E5', scale_factor=0.7)
        self.play(Create(rect))
        self.wait(1)
