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
        self.setup_layout("The Definition of Independence", [
            "Independence means events don't influence each other.",
            "P(A|B) equals P(A) if A and B are independent.",
            "Knowing B gives zero new information about A."
        ])

        # Assets
        coin_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg")
        dice_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/dice.svg")

        # Objects for animations
        circle_a = Circle(radius=0.7, color=BLUE).set_fill(BLUE, opacity=0.3)
        circle_b = Circle(radius=0.7, color=RED).set_fill(RED, opacity=0.3)
        label_a = MathTex("A").move_to(circle_a.get_center())
        label_b = MathTex("B").move_to(circle_b.get_center())
        group_ab = VGroup(circle_a, circle_b, label_a, label_b, coin_icon)
        
        formula_prob = MathTex("P(A \\cap B) = P(A)P(B)", color=WHITE)
        formula_indep = MathTex("P(A|B) = P(A)", color=WHITE)
        box = VGroup(SurroundingRectangle(formula_indep, color=YELLOW, buff=0.1), dice_icon)

        # === Animation for Lecture Line 1 ===
        # Independence means events don't influence each other.
        self.place_in_area(group_ab, 'E4', 'F6', scale_factor=0.6)
        self.play(FadeIn(group_ab))
        self.lecture[0].set_color(BLUE)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # P(A|B) equals P(A) if A and B are independent.
        self.place_at_grid(formula_prob, 'B3', scale_factor=0.9)
        self.play(Write(formula_prob))
        self.play(formula_prob.animate.set_color(WHITE))
        self.play(Indicate(formula_prob))
        
        self.place_at_grid(formula_indep, 'C3', scale_factor=0.9)
        self.play(Write(formula_indep))
        self.play(formula_indep.animate.set_color("#FF00FF"))
        self.lecture[1].set_color("#FF00FF")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Knowing B gives zero new information about A.
        self.place_at_grid(box, 'D3', scale_factor=0.8)
        self.play(Create(box))
        self.lecture[2].set_color(YELLOW)
        self.wait(2)
