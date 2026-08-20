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
            "Independent events make Bayes' theorem collapse.",
            "If independent, P(A|B) simplifies to P(A).",
            "New evidence becomes irrelevant to our prediction.",
            "Example: Unrelated rain and weather apps.",
            "Independence means no Bayesian update occurs."
        ]
        self.setup_layout("Independence vs. Bayes' Theorem", lecture_lines)
        
        # Define elements
        colors = [BLUE_A, GREEN_A, YELLOW_A, RED_A, ORANGE]
        
        # Bayes equation parts
        bayes_eq = MathTex("P(A|B) = \\frac{P(B|A)P(A)}{P(B)}", font_size=32)
        indep_eq = MathTex("P(A|B) = P(A)", font_size=36, color=YELLOW)
        
        # Assets
        rain_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rain.svg")
        phone_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/phone.svg")
        
        # Visual representation
        circle_a = Circle(radius=0.8, color=BLUE).set_fill(BLUE, opacity=0.3)
        circle_b = Circle(radius=0.8, color=GREEN).set_fill(GREEN, opacity=0.3)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(colors[0])
        self.place_in_area(bayes_eq, "A2", "B5", scale_factor=1.0)
        self.play(Write(bayes_eq))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(colors[1])
        self.play(FadeOut(bayes_eq))
        self.place_in_area(indep_eq, "A2", "B5", scale_factor=1.0)
        self.place_at_grid(rain_icon, "C3", scale_factor=0.5)
        self.play(Create(indep_eq), FadeIn(rain_icon))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(colors[2])
        # Demonstrate independence with disjoint circles
        self.place_at_grid(circle_a, "D2", scale_factor=0.6)
        self.place_at_grid(circle_b, "D3", scale_factor=0.6)
        self.place_at_grid(phone_icon, "D6", scale_factor=0.5)
        self.play(DrawBorderThenFill(circle_a), DrawBorderThenFill(circle_b), FadeIn(phone_icon))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(colors[3])
        text_ex = Text("Rain (A)  |  App (B)", font_size=24, color=WHITE)
        self.place_at_grid(text_ex, "E2", scale_factor=0.7)
        self.play(FadeIn(text_ex))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(colors[4])
        # Highlight independence visually
        self.play(Indicate(indep_eq), Indicate(circle_a), Indicate(circle_b))
        self.wait(2)
