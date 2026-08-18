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

class Section2Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "We define limits using epsilon and delta.",
            "Epsilon sets the vertical target tolerance.",
            "Delta restricts the horizontal input range."
        ]
        self.setup_layout("The Rigorous Foundation: Epsilon-Delta Definition", lecture_lines)
        
        # Create objects
        ineq = MathTex(r"|f(x) - L| < \epsilon", font_size=32, color=WHITE)
        target_strip = Rectangle(height=2.0, width=4.0, color="#FF4500", fill_opacity=0.3)
        input_strip = Rectangle(height=4.0, width=1.0, color="#32CD32", fill_opacity=0.3)

        # Asset loading placeholders (none.svg does not exist, using standard mobjects as fallback per instructions)
        # Placeholder for visual asset logic
        
        # === Animation for Lecture Line 1 ===
        self.place_at_grid(ineq, 'A3', scale_factor=1.0)
        self.play(Write(ineq))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))

        # === Animation for Lecture Line 2 ===
        self.place_in_area(target_strip, 'C3', 'D4', scale_factor=0.8)
        self.play(FadeIn(target_strip))
        self.play(self.lecture[1].animate.set_color("#FF4500"))

        # === Animation for Lecture Line 3 ===
        self.place_in_area(input_strip, 'B4', 'E4', scale_factor=0.8)
        self.play(FadeIn(input_strip))
        self.play(self.lecture[2].animate.set_color("#32CD32"))
        
        self.wait(2)
