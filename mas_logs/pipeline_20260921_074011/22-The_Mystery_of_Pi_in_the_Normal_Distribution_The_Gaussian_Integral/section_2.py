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
        self.setup_layout("Prerequisite: The Gaussian Integral", ["Consider the Gaussian integral, I.", "Its anti-derivative is not elementary.", "We need a clever trick to solve it."])
        
        # === Animation for Lecture Line 1 ===
        # Load asset /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg
        asset_1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg", should_center=False)
        self.place_at_grid(asset_1, 'A3', scale_factor=0.5)
        
        integral_expr = MathTex(r"I = \int_{-\infty}^{\infty} e^{-x^2} dx", color="#FFFFFF")
        self.place_at_grid(integral_expr, 'B3', scale_factor=1.0)
        self.play(Write(integral_expr), FadeIn(asset_1))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Write I² = (∫ e^(-x²) dx) (∫ e^(-y²) dy)
        sq_integral_expr = MathTex(r"I^2 = \left(\int_{-\infty}^{\infty} e^{-x^2} dx\right) \left(\int_{-\infty}^{\infty} e^{-y^2} dy\right)", color="#FFFF00")
        self.place_at_grid(sq_integral_expr, 'D3', scale_factor=0.8)
        self.play(Write(sq_integral_expr))
        self.lecture[1].set_color("#FFFF00")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Write I² = ∫ ∫ e^(-(x² + y²)) dx dy
        # Load asset /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg
        asset_2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg", should_center=False)
        self.place_at_grid(asset_2, 'F3', scale_factor=0.5)
        
        double_integral_expr = MathTex(r"I^2 = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} e^{-(x^2+y^2)} dx dy", color="#00FF00")
        self.place_in_area(double_integral_expr, 'E2', 'F5', scale_factor=0.9)
        self.play(ReplacementTransform(sq_integral_expr, double_integral_expr), FadeIn(asset_2))
        self.lecture[2].set_color("#00FF00")
        self.wait(1)
