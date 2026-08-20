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
        self.setup_layout("Defining Pi: The Mathematical Formalization", [
            "Pi is the circle's circumference divided by diameter.",
            "It is an irrational, endless decimal number.",
            "The sequence of digits never repeats."
        ])
        
        # Elements
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg
        circle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg", color=BLUE)
        self.place_at_grid(circle, 'C2', scale_factor=1.0)
        
        diameter_line = Line(circle.get_left(), circle.get_right(), color=YELLOW)
        c_label = Text("C", color=BLUE).next_to(circle, UP)
        d_label = Text("d", color=YELLOW).next_to(diameter_line, DOWN)
        
        formula = MathTex(r"\\pi = \\frac{C}{d}", color=WHITE)
        self.place_in_area(formula, 'C4', 'C5', scale_factor=0.9)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(FadeIn(circle), Create(diameter_line))
        self.play(Write(c_label), Write(d_label))
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(YELLOW))
        pi_val = MathTex(r"\\pi \\approx 3.14159...", color=WHITE)
        self.place_at_grid(pi_val, 'D4', scale_factor=0.8)
        self.play(FadeIn(pi_val))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(YELLOW))
        dots = Text("...", font_size=36, color=WHITE)
        self.place_at_grid(dots, 'E4', scale_factor=0.8)
        self.play(FadeIn(dots))
        self.wait(2)
