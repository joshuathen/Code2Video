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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "First, identify your new basis vectors.",
            "Second, build matrix P using those vectors.",
            "Third, multiply P by your coordinate vector.",
            "This maps to standard representation efficiently.",
            "Procedure converts any basis to standard coordinates."
        ]
        self.setup_layout("Step-by-Step Calculation Procedure", lecture_lines)
        
        # Prepare Mobjects
        P = Matrix([[2, 1], [1, 2]], left_bracket="[", right_bracket="]")
        V = Matrix([[1], [1]], left_bracket="[", right_bracket="]")
        V_prime = Matrix([[3], [3]], left_bracket="[", right_bracket="]")
        
        equation = MathTex("P", " \\cdot ", "V", "=", "V'").scale(1.2)
        
        calculator_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg")
        computer_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg")
        
        arrow = Arrow(start=ORIGIN, end=RIGHT*1.5, color=WHITE)
        
        # Positioning based on VideoCritic instructions
        self.place_at_grid(P, "B2", scale_factor=0.7)
        self.place_at_grid(V, "E2", scale_factor=0.7)
        self.place_in_area(equation, "D2", "D5", scale_factor=0.6)
        
        # Set opacity for entrance
        for mob in [P, V, equation, arrow, calculator_icon, computer_icon, V_prime]:
            mob.set_opacity(0)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.play(FadeIn(P))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(BLUE))
        self.play(FadeIn(V))
        self.place_at_grid(calculator_icon, "B5", scale_factor=0.5)
        self.play(FadeIn(calculator_icon))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(BLUE))
        self.place_at_grid(arrow, "C3", scale_factor=1)
        self.play(FadeIn(equation), FadeIn(arrow))
        
        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(BLUE))
        self.place_at_grid(computer_icon, "E5", scale_factor=0.5)
        self.play(FadeIn(computer_icon), FadeIn(V_prime.set_color(YELLOW).move_to(self.grid["F5"])))
        
        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(BLUE))
        self.play(Flash(V_prime, color=WHITE))
        self.wait(1)
