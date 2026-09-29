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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Derivatives and integrals are inverse processes.",
            "Two sides of one beautiful coin.",
            "They bridge snapshots and dynamic motion.",
            "This is the fundamental theorem of calculus.",
            "Understanding change gives us the world."
        ]
        self.setup_layout("Fundamental Theorem of Calculus", lecture_lines)
        
        # Animation Elements
        coin = Circle(radius=1, color=YELLOW, fill_opacity=0.5)
        text_der = Text("Derivative", font_size=24, color=BLUE)
        text_int = Text("Integral", font_size=24, color=RED)
        
        # Pre-positioning
        self.place_in_area(coin, 'B2', 'C4', scale_factor=0.5)
        text_der.next_to(coin, UP)
        text_int.next_to(coin, DOWN)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.play(Create(coin), Write(text_der), Write(text_int))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.play(Rotate(coin, angle=PI))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN))
        snapshot = Square(side_length=1.5, color=WHITE).shift(RIGHT * 3)
        motion = TracedPath(coin.get_center, stroke_color=GREEN)
        self.play(Create(snapshot))
        self.add(motion)
        self.play(coin.animate.shift(RIGHT * 3))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(PURPLE))
        theorem = Text("FTC", font_size=40, color=PURPLE)
        self.place_at_grid(theorem, 'E6', scale_factor=0.7)
        self.play(Write(theorem))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(ORANGE))
        self.wait(2)
