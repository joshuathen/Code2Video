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
        self.setup_layout("The Roots of Unity Filter Technique", [
            "The filter identity uses averages of P(ω^k).",
            "Evaluating at roots filters non-multiple indices.",
            "This leaves only a0, an, a2n, ... terms.",
            "We extract specific sums from binomial expansions.",
            "The filter isolates target subsets elegantly."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Display P(ω) evaluation in #FFFFFF
        p_omega = MathTex(r"P(\omega) = \sum a_k \omega^k", color=WHITE)
        self.place_at_grid(p_omega, 'B2', scale_factor=1.0)
        self.play(Write(p_omega))
        self.lecture[0].set_color(WHITE)

        # === Animation for Lecture Line 2 ===
        # Fade in omega powers (ω^0, ω^1,...) in #FFA500
        omega_powers = VGroup(*[MathTex(r"\omega^{%d}" % k, color="#FFA500") for k in range(4)])
        omega_powers.arrange(RIGHT, buff=0.3)
        self.place_at_grid(omega_powers, 'C3', scale_factor=0.8)
        self.play(FadeIn(omega_powers))
        self.lecture[1].set_color("#FFA500")

        # === Animation for Lecture Line 3 ===
        # Filter coefficients by darkening others in #404040 [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/filter.svg]
        filter_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/filter.svg")
        self.place_at_grid(filter_icon, 'E2', scale_factor=0.5)
        self.play(FadeIn(filter_icon))
        # Logic representation
        coefs = MathTex(r"a_0, a_n, a_{2n}, \dots", color="#404040")
        self.place_at_grid(coefs, 'E4', scale_factor=0.8)
        self.play(Write(coefs))
        self.lecture[2].set_color("#404040")

        # === Animation for Lecture Line 4 ===
        # Show binomial expansion P(x) = (1+x)^n in #00FF00
        binom = MathTex(r"P(x) = (1+x)^n", color="#00FF00")
        self.place_at_grid(binom, 'B5', scale_factor=0.8)
        self.play(Write(binom))
        self.lecture[3].set_color("#00FF00")

        # === Animation for Lecture Line 5 ===
        # Final sum result appears in #FFFF00
        result = MathTex(r"\sum \binom{n}{3k} = \dots", color="#FFFF00")
        self.place_at_grid(result, 'D5', scale_factor=0.8)
        self.play(Write(result))
        self.lecture[4].set_color("#FFFF00")
        self.wait(2)
