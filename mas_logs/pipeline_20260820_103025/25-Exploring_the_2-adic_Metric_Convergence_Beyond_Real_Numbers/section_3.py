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
            "Series converge if terms shrink toward zero.",
            "In 2-adic, series exhibit ultrametric behavior.",
            "1, 2, 4, 8 sequence converges to zero."
        ]
        self.setup_layout("Convergence: A Different Perspective", lecture_lines)
        
        # Load assets
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")

        # === Animation for Lecture Line 1 ===
        line1_text = self.lecture[0]
        line1_text.set_color(YELLOW)
        
        real_line = NumberLine(x_range=[0, 10, 2], length=4, color=BLUE)
        real_label = Text("Real Space", font_size=18, color=BLUE)
        adic_line = NumberLine(x_range=[0, 10, 2], length=4, color=RED)
        adic_label = Text("2-adic Space", font_size=18, color=RED)
        
        # Applying fix from Issue 28/43
        space_labels = VGroup(real_label, real_line, adic_label, adic_line)
        self.place_in_area(space_labels, 'A2', 'B5', scale_factor=0.75)
        
        self.place_at_grid(ruler, "C2", scale_factor=0.5)
        
        self.play(Create(real_line), Write(real_label), FadeIn(ruler))
        self.play(Create(adic_line), Write(adic_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        line1_text.set_color(WHITE)
        line2_text = self.lecture[1]
        line2_text.set_color(YELLOW)
        
        formula = MathTex(r"|x+y|_2 \leq \max(|x|_2, |y|_2)", font_size=24)
        
        # Applying fix from Issue 26/41
        self.place_in_area(formula, 'E2', 'F5', scale_factor=0.9)
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        line2_text.set_color(WHITE)
        line3_text = self.lecture[2]
        line3_text.set_color(YELLOW)
        
        sequence_text = Text("1, 2, 4, 8...", font_size=24, color=GREEN)
        # Applying fix from Issue 27/42
        self.place_at_grid(sequence_text, 'B3', scale_factor=0.85)
        
        sequence_dots = VGroup(*[Dot(color=GREEN) for _ in range(4)])
        for i, dot in enumerate(sequence_dots):
            self.place_at_grid(dot, f"B{i+3}")
        
        limit_point = Circle(radius=0.2, color=YELLOW).move_to(self.grid["F5"])
        
        self.place_at_grid(compass, "D5", scale_factor=0.5)
        
        self.play(Write(sequence_text), FadeIn(compass))
        self.play(LaggedStart(*[FadeIn(d) for d in sequence_dots], lag_ratio=0.5))
        self.play(FadeIn(limit_point), Flash(limit_point))
        self.wait(2)
