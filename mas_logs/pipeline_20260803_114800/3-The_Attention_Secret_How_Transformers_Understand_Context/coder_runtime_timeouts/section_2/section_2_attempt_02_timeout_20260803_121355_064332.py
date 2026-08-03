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
        # Initial highlighting
        self.lecture[0].set_color(WHITE)
        
        # Draw light gray (#444444) grid lines.
        grid_lines = VGroup()
        for r in ["A", "B", "C", "D", "E", "F"]:
            grid_lines.add(Line(self.grid[f"{r}1"], self.grid[f"{r}6"], color="#444444", stroke_width=1, stroke_opacity=0.5))
        for c in ["1", "2", "3", "4", "5", "6"]:
            grid_lines.add(Line(self.grid[f"A{c}"], self.grid[f"F{c}"], color="#444444", stroke_width=1, stroke_opacity=0.5))
        
        # 1. Display words 'King' [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/king.svg], 'Queen', and 'Apple'
        # King icon from SVG asset
        king_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/king.svg").set_color(WHITE)
        self.place_at_grid(king_icon, "B1", scale_factor=0.35)
        king_label = Text("King", font_size=18, color=WHITE)
        king_label.next_to(king_icon, UP, buff=0.1)
        
        # Initial Queen position set to B4 to ensure 'too far' initially then 'move closer' later
        queen_dot = Dot(color=WHITE)
        self.place_at_grid(queen_dot, "B4", scale_factor=1.0)
        queen_label = Text("Queen", font_size=18, color=WHITE)
        queen_label.next_to(queen_dot, UP, buff=0.1)
        
        apple_dot = Dot(color=WHITE)
        self.place_at_grid(apple_dot, "E2", scale_factor=1.0)
        apple_label = Text("Apple", font_size=18, color=WHITE)
        apple_label.next_to(apple_dot, UP, buff=0.1)
        
        self.play(Create(grid_lines), run_time=1)
        self.play(
            FadeIn(king_icon), FadeIn(king_label),
            FadeIn(queen_dot), FadeIn(queen_label),
            FadeIn(apple_dot), FadeIn(apple_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Transition: Highlight line 2 in Cyan
        self.play(self.lecture[1].animate.set_color("#00FFFF"), run_time=0.5)
        
        # 2. Move 'King' and 'Queen' points closer together while changing color to cyan (#00FFFF). 
        # Target positions: B2 for King, B3 for Queen (resolves Issue 20)
        target_king_pos = self.grid["B2"]
        target_queen_pos = self.grid["B3"]
        
        self.play(
            king_icon.animate.move_to(target_king_pos).set_color("#00FFFF"),
            king_label.animate.next_to(target_king_pos, UP, buff=0.1).set_color("#00FFFF"),
            queen_dot.animate.move_to(target_queen_pos).set_color("#00FFFF"),
            queen_label.animate.next_to(target_queen_pos, UP, buff=0.1).set_color("#00FFFF"),
            apple_dot.animate.set_color("#FF0000"),
            apple_label.animate.set_color("#FF0000"),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Transition: Highlight line 3 in Purple
        self.play(self.lecture[2].animate.set_color("#A020F0"), run_time=0.5)
        
        # 3. Introduce the word 'Computer' (#A020F0).
        computer_dot = Dot(color="#A020F0")
        self.place_at_grid(computer_dot, "D5", scale_factor=1.0)
        computer_label = Text("Computer", font_size=18, color="#A020F0")
        computer_label.next_to(computer_dot, UP, buff=0.1)
        
        self.play(FadeIn(computer_dot), FadeIn(computer_label))
        
        # Vector arrow from Apple to Computer (resolves Issue 21 via updater)
        arrow = Arrow(apple_dot.get_center(), computer_dot.get_center(), buff=0.1, color="#A020F0", stroke_width=4)
        # Updater ensures the arrow follows the dot as it slides
        arrow.add_updater(lambda m: m.put_start_and_end_on(apple_dot.get_center(), computer_dot.get_center()))
        
        self.play(GrowArrow(arrow))
        
        # Apple slides toward Computer. Target: E5 (resolves Issue 19 and 21)
        target_apple_pos = self.grid["E5"]
        self.play(
            apple_dot.animate.move_to(target_apple_pos),
            apple_label.animate.next_to(target_apple_pos, UP, buff=0.1),
            run_time=2
        )
        self.wait(3)
