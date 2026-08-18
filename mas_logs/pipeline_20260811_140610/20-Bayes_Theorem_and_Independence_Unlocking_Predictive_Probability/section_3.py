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
        self.setup_layout("Introducing Bayes' Theorem: The Logic of Inference", [
            "Bayes' Theorem updates beliefs with new evidence.",
            "Starts with a prior belief about an event.",
            "Multiplies by likelihood to get the posterior.",
            "This process refines our understanding of probability.",
            "It turns initial guesses into calculated conclusions."
        ])
        
        # Hide lecture initially
        for line in self.lecture:
            line.set_opacity(0)

        # Load assets
        calculator = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg")
        notebook = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/notebook.svg")

        equation = MathTex(
            r"P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}",
            font_size=48
        )
        
        # Color-coding components
        # Prior P(A) -> #3357FF (BLUE)
        # Likelihood P(B|A) -> #FF33A8 (PINK)
        equation.set_color_by_tex("P(A)", "#3357FF")
        equation.set_color_by_tex("P(B|A)", "#FF33A8")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_opacity(1)
        self.place_at_grid(calculator, "B6", scale_factor=0.3)
        self.place_in_area(equation, 'A2', 'C5', scale_factor=1.2)
        self.play(FadeIn(calculator), Write(equation))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_opacity(1)
        prior_label = Text("Prior: P(A)", color="#3357FF", font_size=24)
        self.place_at_grid(prior_label, 'D3', scale_factor=0.7)
        self.play(FadeIn(prior_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_opacity(1)
        like_label = Text("Likelihood: P(B|A)", color="#FF33A8", font_size=24)
        self.place_at_grid(like_label, 'D4', scale_factor=0.7)
        self.play(FadeIn(like_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_opacity(1)
        posterior_label = Text("Posterior: P(A|B)", color=WHITE, font_size=24)
        self.place_at_grid(posterior_label, 'E4', scale_factor=0.7)
        self.play(FadeIn(posterior_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_opacity(1)
        self.place_at_grid(notebook, "F6", scale_factor=0.3)
        self.play(
            FadeIn(notebook),
            Indicate(equation),
            run_time=2
        )
        self.wait(2)
