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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Conclusion: Complexity and Efficiency", [
            "The puzzle is a physical binary counter.",
            "Moves grow exponentially: two to n minus one.",
            "Math dictates the movement of objects."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Represent Binary Counter: 1111 (for 4 disks)
        binary_display = VGroup(
            Text("1", color=BLUE),
            Text("1", color=BLUE),
            Text("1", color=BLUE),
            Text("1", color=BLUE)
        ).arrange(RIGHT, buff=0.3)
        self.place_at_grid(binary_display, 'C3', scale_factor=1.0)
        self.play(FadeIn(binary_display))
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        # Exponential Growth: 2^n - 1
        formula = MathTex("2^n - 1", color=GREEN).scale(1.5)
        self.place_at_grid(formula, 'D3', scale_factor=1.2)
        self.play(Write(formula))
        self.lecture[1].set_color(GREEN)

        # === Animation for Lecture Line 3 ===
        # Final visualization: 15 moves representation (map)
        move_label = Text("15 Moves", color=YELLOW)
        self.place_at_grid(move_label, 'E3', scale_factor=0.9)
        self.play(Write(move_label))
        self.lecture[2].set_color(YELLOW)
        self.wait(2)
