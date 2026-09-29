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
        lecture_lines = [
            "Explicit functions are like a neatly folded shirt.",
            "Implicit relations are like a tangled ball of yarn.",
            "Recall the Chain Rule for differentiation."
        ]
        self.setup_layout("Prerequisite Review: Explicit vs. Implicit", lecture_lines)
        
        explicit_text = Text("Explicit: y = f(x)", color=BLUE)
        implicit_text = Text("Implicit: f(x, y) = 0", color=RED)
        
        shirt = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/shirt.svg")
        yarn = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/yarn.svg")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.place_in_area(explicit_text, 'A1', 'B6', scale_factor=0.6)
        self.place_at_grid(shirt, 'B3', scale_factor=0.5)
        self.play(FadeIn(explicit_text), FadeIn(shirt))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(RED)
        self.place_in_area(implicit_text, 'C1', 'D6', scale_factor=0.6)
        self.place_at_grid(yarn, 'D3', scale_factor=0.5)
        self.play(FadeIn(implicit_text), FadeIn(yarn))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        chain_rule = MathTex(r"\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)", color=YELLOW)
        self.place_at_grid(chain_rule, 'E2', scale_factor=0.7)
        self.play(FadeIn(chain_rule))
        
        self.wait(2)
