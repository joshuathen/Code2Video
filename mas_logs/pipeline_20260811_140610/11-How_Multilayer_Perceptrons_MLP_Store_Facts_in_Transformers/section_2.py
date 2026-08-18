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
        self.setup_layout("The MLP as a Key-Value Associative Memory", [
            "MLP layers are key-value associative memories.",
            "First layer acts as a pattern matcher.",
            "Second layer retrieves specific fact outputs."
        ])
        
        # Asset loading
        input_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/key.svg", color="#3498DB")
        output_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/database.svg", color="#2ECC71")

        # Setup visuals
        input_rect = Rectangle(width=2, height=0.5, color="#3498DB").set_fill(opacity=0.5)
        input_label = Text("Input Query", font_size=20)
        input_group = VGroup(input_rect, input_label, input_icon)
        input_icon.next_to(input_rect, UP, buff=0.1).scale(0.7)
        self.place_at_grid(input_group, "B3", scale_factor=0.8)
        
        key_triangle = Triangle(color="#E74C3C").set_fill(opacity=0.5)
        key_label = Text("Pattern (Key)", font_size=20)
        key_group = VGroup(key_triangle, key_label)
        self.place_in_area(key_group, "C4", "D5", scale_factor=0.6)
        
        output_rect = Rectangle(width=2, height=0.5, color="#2ECC71").set_fill(opacity=0.5)
        output_label = Text("Fact (Value)", font_size=20)
        output_group = VGroup(output_rect, output_label, output_icon)
        output_icon.next_to(output_rect, UP, buff=0.1).scale(0.7)
        self.place_at_grid(output_group, "E4", scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_opacity(1), run_time=0.5)
        self.play(FadeIn(input_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_opacity(1), run_time=0.5)
        self.play(FadeIn(key_group))
        arrow1 = Arrow(start=input_group.get_bottom(), end=key_group.get_top(), color=WHITE)
        self.play(Create(arrow1))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_opacity(1), run_time=0.5)
        self.play(FadeIn(output_group))
        arrow2 = Arrow(start=key_group.get_bottom(), end=output_group.get_top(), color=WHITE)
        self.play(Create(arrow2))
        self.wait(1)
