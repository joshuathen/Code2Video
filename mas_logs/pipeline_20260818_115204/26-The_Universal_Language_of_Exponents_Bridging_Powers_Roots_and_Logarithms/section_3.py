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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Logarithms act as detectives.",
            "They hunt for missing exponents.",
            "Log_b(x) = y finds the exponent.",
            "Example: 3^4 = 81 means log_3(81) = 4.",
            "Logs bridge powers and roots."
        ]
        self.setup_layout("Logarithms: Seeking the Exponent", lecture_lines)
        
        # Assets
        detective = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/detective.svg")
        magnifying_glass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifyingglass.svg")
        
        # Elements
        log_eq = MathTex(r"\\log_b(y) = x", font_size=48, color=WHITE)
        pow_eq = MathTex(r"b^y = x", font_size=48)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.place_at_grid(detective, 'A4', scale_factor=0.5)
        self.place_at_grid(log_eq, 'B4', scale_factor=1.0)
        self.play(FadeIn(detective), Write(log_eq))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        highlight_x = Circle(color=YELLOW, radius=0.3).move_to(log_eq[0][4])
        self.play(Create(highlight_x))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(ORANGE)
        self.play(FadeOut(highlight_x))
        self.place_at_grid(pow_eq, 'C4', scale_factor=1.0)
        self.play(Write(pow_eq))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(RED)
        example = MathTex(r"3^4 = 81", font_size=36)
        self.place_at_grid(example, 'E4', scale_factor=0.9)
        self.play(Write(example))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(PURPLE)
        final_eq = MathTex(r"\\log_3(81) = 4", font_size=48, color=WHITE)
        self.place_at_grid(magnifying_glass, 'D2', scale_factor=0.5)
        self.place_at_grid(final_eq, 'E4', scale_factor=1.0)
        self.play(FadeOut(log_eq), FadeOut(pow_eq), FadeOut(example), FadeIn(magnifying_glass), Write(final_eq))
        self.wait(2)
