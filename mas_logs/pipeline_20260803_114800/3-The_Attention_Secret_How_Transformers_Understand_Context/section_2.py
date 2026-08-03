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
        title_text = "Prerequisite: Turning Words into Space"
        lecture_lines = [
            "Computers first convert words into lists of numbers.",
            "These vectors represent words as points in space.",
            "Similar meanings sit closer together in this system."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Initial state highlight
        self.lecture[0].set_color(WHITE)
        
        # Grid lines (light gray #444444)
        grid_lines = VGroup(*[
            Line(self.grid[f"{r}1"], self.grid[f"{r}6"], color="#444444", stroke_width=1, stroke_opacity=0.5)
            for r in ["A", "B", "C", "D", "E", "F"]
        ] + [
            Line(self.grid[f"A{c}"], self.grid[f"F{c}"], color="#444444", stroke_width=1, stroke_opacity=0.5)
            for c in ["1", "2", "3", "4", "5", "6"]
        ])
        
        # Display words 'King' [Asset], 'Queen', and 'Apple'
        king_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/king.svg").set_color(WHITE)
        self.place_at_grid(king_icon, "B1", scale_factor=0.3)
        king_label = Text("King", font_size=16, color=WHITE).next_to(king_icon, UP, buff=0.1)
        
        # Queen point - closer than previous B5 to illustrate semantic relationship (Issue 20)
        queen_dot = Dot(color=WHITE)
        self.place_at_grid(queen_dot, "B4", scale_factor=0.8)
        queen_label = Text("Queen", font_size=16, color=WHITE).next_to(queen_dot, UP, buff=0.1)
        
        # Apple point
        apple_dot = Dot(color=WHITE)
        self.place_at_grid(apple_dot, "E2", scale_factor=1.0)
        apple_label = Text("Apple", font_size=16, color=WHITE).next_to(apple_dot, UP, buff=0.1)
        
        self.play(
            Create(grid_lines),
            FadeIn(king_icon), FadeIn(king_label),
            FadeIn(queen_dot), FadeIn(queen_label),
            FadeIn(apple_dot), FadeIn(apple_label),
            run_time=1.0
        )
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        highlight_color_2 = "#00FFFF" # Cyan
        self.play(self.lecture[1].animate.set_color(highlight_color_2), run_time=0.4)
        
        # Move 'King' and 'Queen' closer together (B2 and B3) while changing color
        target_king_pos = self.grid["B2"]
        target_queen_pos = self.grid["B3"]
        
        self.play(
            king_icon.animate.move_to(target_king_pos).set_color(highlight_color_2),
            king_label.animate.next_to(target_king_pos, UP, buff=0.1).set_color(highlight_color_2),
            queen_dot.animate.move_to(target_queen_pos).set_color(highlight_color_2),
            queen_label.animate.next_to(target_queen_pos, UP, buff=0.1).set_color(highlight_color_2),
            apple_dot.animate.set_color("#FF0000"), # Apple remains distant (Red)
            apple_label.animate.set_color("#FF0000"),
            run_time=1.2
        )
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        highlight_color_3 = "#A020F0" # Purple
        self.play(self.lecture[2].animate.set_color(highlight_color_3), run_time=0.4)
        
        # Introduce 'Computer'
        computer_dot = Dot(color=highlight_color_3)
        self.place_at_grid(computer_dot, "D5", scale_factor=1.0)
        computer_label = Text("Computer", font_size=16, color=highlight_color_3).next_to(computer_dot, UP, buff=0.1)
        
        self.play(FadeIn(computer_dot), FadeIn(computer_label), run_time=0.6)
        
        # Purple vector arrow pulls Apple
        connecting_arrow = Arrow(
            apple_dot.get_center(), 
            computer_dot.get_center(), 
            buff=0.1, 
            color=highlight_color_3, 
            stroke_width=3
        )
        # Tracking updater
        connecting_arrow.add_updater(lambda m: m.put_start_and_end_on(apple_dot.get_center(), computer_dot.get_center()))
        
        self.play(Create(connecting_arrow), run_time=0.8)
        
        # Apple slides toward Computer. Target: E5 to avoid overlap with computer_dot at D5 (Issue 19 & 21)
        target_apple_pos = self.grid["E5"]
        self.play(
            apple_dot.animate.move_to(target_apple_pos),
            apple_label.animate.next_to(target_apple_pos, UP, buff=0.1),
            run_time=1.2
        )
        
        connecting_arrow.clear_updaters()
        self.wait(1.5)
