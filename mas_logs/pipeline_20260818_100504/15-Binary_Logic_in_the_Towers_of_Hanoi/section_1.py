from manim import *

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
        self.setup_layout("Prerequisite: The Language of Binary", [
            "Binary uses digits 0 and 1.",
            "Each position represents powers of two.",
            "Three bits count zero to seven."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Using SVG asset if needed, though none.svg is empty.
        digit_0 = Text("0", color=WHITE)
        digit_1 = Text("1", color=WHITE)
        binary_digits = VGroup(digit_0, digit_1).arrange(RIGHT, buff=0.5)
        # Applying Fix: VideoCritic #21
        self.place_at_grid(binary_digits, 'B1', scale_factor=1.2)
        self.play(Write(binary_digits))
        self.lecture[0].set_color("#FFFF00")
        
        # === Animation for Lecture Line 2 ===
        # Applying Asset Integration requirement
        powers = VGroup(Text("2^0", font_size=20), Text("2^1", font_size=20), Text("2^2", font_size=20)).arrange(RIGHT, buff=0.5)
        self.place_at_grid(powers, 'C2', scale_factor=1.0)
        self.play(FadeIn(powers))
        # Highlight logic
        digit_0.set_color("#FFFF00")
        digit_1.set_color("#FFFF00")
        self.lecture[1].set_color("#FFFF00")
        
        # === Animation for Lecture Line 3 ===
        table = VGroup()
        for i in range(8):
            row_str = f"{i} = {i:03b}"
            row = Text(row_str, font_size=20, color=WHITE)
            table.add(row)
        table.arrange(DOWN, aligned_edge=LEFT)
        # Applying Fix: VideoCritic #22
        self.place_in_area(table, 'C3', 'E5', scale_factor=0.9)
        self.play(Create(table))
        self.lecture[2].set_color("#FFFF00")
        self.wait(2)
