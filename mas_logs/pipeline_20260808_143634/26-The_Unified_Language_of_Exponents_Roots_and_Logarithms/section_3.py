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
        self.setup_layout(
            "Logarithms: Searching for the Time/Rate", 
            ["Logarithms solve for the exponent.", "How long to reach a value?", "The base is fixed here.", "log_b(y) equals x exactly.", "Logs find the growth time."]
        )
        
        # Elements
        base_eq = MathTex("3^x = 81", font_size=40, color=YELLOW)
        q_mark = Tex("?", font_size=60, color=RED)
        log_eq = MathTex("\\log_3(81) = x", font_size=40, color=BLUE)
        
        # Assets
        calculator = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg")
        magnifying_glass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifyingglass.svg")

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.lecture[0]))
        self.place_in_area(base_eq, 'B2', 'B3', scale_factor=1.0)
        self.play(Write(base_eq))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(self.lecture[1]))
        self.place_at_grid(q_mark, 'B4', scale_factor=0.8)
        self.play(FadeIn(q_mark))
        self.lecture[1].set_color(RED)

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(self.lecture[2]))
        base_rect = SurroundingRectangle(base_eq, color=YELLOW)
        self.place_at_grid(calculator, 'E3', scale_factor=0.5)
        self.play(Create(base_rect), FadeIn(calculator))
        self.lecture[2].set_color(YELLOW)

        # === Animation for Lecture Line 4 ===
        self.play(FadeIn(self.lecture[3]))
        self.place_in_area(log_eq, 'C2', 'C4', scale_factor=1.1)
        self.play(FadeOut(q_mark), FadeOut(base_rect), FadeOut(calculator), Write(log_eq))
        self.lecture[3].set_color(BLUE)

        # === Animation for Lecture Line 5 ===
        self.play(FadeIn(self.lecture[4]))
        self.place_at_grid(magnifying_glass, 'E4', scale_factor=0.5)
        highlight = SurroundingRectangle(log_eq, color=BLUE)
        self.play(Create(highlight), FadeIn(magnifying_glass))
        self.lecture[4].set_color(BLUE)
        
        self.wait(2)
