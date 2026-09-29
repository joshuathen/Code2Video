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
            "L'Hôpital's Rule: The Shortcut for Indeterminacy",
            ["L'Hôpital solves indeterminate 0/0 forms.",
             "Derivatives compare the growth rates of functions.",
             "Slope comparison reveals the limit's value.",
             "Use this shortcut when algebra fails us.",
             "Growth rates converge to the limit ratio."]
        )
        
        # Animations
        # === Animation for Lecture Line 1 ===
        # Show an indeterminate 0/0 expression
        calc_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg")
        expr = MathTex(r"\lim_{x \to 0} \frac{\sin(x)}{x} = \frac{0}{0}", color="#FFFFFF")
        self.place_at_grid(calc_icon, 'A4', scale_factor=0.3)
        self.place_at_grid(expr, 'B4', scale_factor=1.0)
        self.play(FadeIn(calc_icon), Write(expr))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Display the derivative fraction setup
        deriv = MathTex(r"\frac{d}{dx} \sin(x) = \cos(x), \quad \frac{d}{dx} x = 1", color="#FF00FF")
        deriv_fraction = MathTex(r"\lim_{x \to 0} \frac{\cos(x)}{1}", color="#FF00FF")
        deriv_group = VGroup(deriv, deriv_fraction).arrange(DOWN)
        self.place_in_area(deriv_group, 'C3', 'D5', scale_factor=0.9)
        self.play(Write(deriv_group))
        self.lecture[1].set_color("#FF00FF")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Animate transformation to a clear limit value
        result = MathTex(r"= \cos(0) = 1", color="#00FFFF")
        self.place_at_grid(result, 'E3', scale_factor=1.1)
        self.play(ReplacementTransform(deriv_fraction.copy(), result))
        self.lecture[2].set_color("#00FFFF")
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Use this shortcut when algebra fails us.
        self.lecture[3].set_color("#FFFF00")
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Growth rates converge to the limit ratio.
        shortcut_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/shortcut.svg")
        self.place_at_grid(shortcut_icon, 'F3', scale_factor=0.3)
        self.play(FadeIn(shortcut_icon))
        self.lecture[4].set_color("#00FF00")
        self.wait(2)
