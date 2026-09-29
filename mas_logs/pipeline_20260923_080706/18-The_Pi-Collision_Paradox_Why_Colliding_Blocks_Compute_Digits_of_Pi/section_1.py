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
        self.setup_layout("The Setup: The Elastic Collision Puzzle", [
            "Two blocks collide on a frictionless surface.",
            "Small block hits a wall.",
            "Blocks exchange momentum elastically.",
            "They keep colliding indefinitely.",
            "This system holds a secret."
        ])
        
        wall = Line(start=UP*1.5, end=DOWN*1.5, color=GRAY).shift(LEFT*2.5)
        floor = Line(start=LEFT*3, end=RIGHT*3, color=WHITE).shift(DOWN*1.5)
        
        small_block = Square(side_length=0.4, color=BLUE, fill_opacity=0.8)
        large_block = Square(side_length=1.0, color=RED, fill_opacity=0.8)
        
        # Initial positions adjusted based on feedback
        self.place_at_grid(small_block, 'E3', scale_factor=0.6)
        self.place_in_area(large_block, 'D4', 'E6', scale_factor=1.0)
        self.add(wall, floor, small_block, large_block)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.play(small_block.animate.shift(RIGHT*1), run_time=1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(GREEN)
        self.play(small_block.animate.shift(LEFT*2), run_time=1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        self.play(small_block.animate.shift(RIGHT*2), large_block.animate.shift(RIGHT*0.5), run_time=1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(ORANGE)
        self.play(small_block.animate.shift(LEFT*1), large_block.animate.shift(RIGHT*0.2), run_time=1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(RED)
        self.wait(1)
