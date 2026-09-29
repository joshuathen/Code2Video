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
        self.setup_layout("The Infinite Sum: Building the Foundation", [
            "The Zeta function sums inverse powers of integers.",
            "For Re(s) > 1, the series converges nicely.",
            "But for s=1, it diverges to infinity."
        ])
        
        # Load asset
        calc_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg")
        self.place_at_grid(calc_icon, 'B1', scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        # The Zeta function sums inverse powers of integers.
        self.play(FadeIn(calc_icon))
        zeta_formula = MathTex(r"\zeta(s) = \sum_{n=1}^{\infty} \frac{1}{n^s}", color="#00FF00")
        self.place_in_area(zeta_formula, 'A2', 'B5', scale_factor=1.2)
        self.play(FadeIn(zeta_formula))
        self.lecture[0].set_color("#00FF00")

        # === Animation for Lecture Line 2 ===
        # For Re(s) > 1, the series converges nicely.
        # Showing terms as partial sums
        series_text = MathTex(r"1 + \frac{1}{2^2} + \frac{1}{3^2} + \dots = \frac{\pi^2}{6} \approx 1.645", color="#FFFF00")
        self.place_at_grid(series_text, 'C2', scale_factor=0.9)
        self.play(FadeIn(series_text))
        self.lecture[1].set_color("#FFFF00")

        # === Animation for Lecture Line 3 ===
        # But for s=1, it diverges to infinity.
        divergence_text = MathTex(r"1 + \frac{1}{2} + \frac{1}{3} + \dots \to \infty", color="#FF0000")
        self.place_at_grid(divergence_text, 'E2', scale_factor=0.9)
        self.play(FadeIn(divergence_text))
        self.lecture[2].set_color("#FF0000")
        
        # Final cleanup
        self.play(FadeOut(calc_icon), FadeOut(zeta_formula), FadeOut(series_text), FadeOut(divergence_text))
        self.wait(2)
