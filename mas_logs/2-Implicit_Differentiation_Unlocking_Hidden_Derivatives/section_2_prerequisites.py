from manim import *

# Removed the explicit TexTemplate configuration to use Manim's default compiler settings,
# which resolves the FileNotFoundError for 'pdflatex' when that specific compiler is missing.

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

class Section2PrerequisitesScene(TeachingScene):
    def construct(self):
        # Initial layout setup
        self.setup_layout(
            "Prerequisite Review: The Chain Rule Power-Up",
            [
                "Remember the chain rule for nested functions.",
                "If y depends on x, y is an inner function.",
                "Differentiating y^2 requires a special 'tag'.",
                "d/dx of y^2 becomes 2y times dy/dx.",
                "This dy/dx is the 'secret sauce' of calculus."
            ]
        )

        # === Animation for Lecture Line 1 ===
        # Display the general Chain Rule in white
        self.lecture[0].set_color(WHITE)
        chain_rule_gen = MathTex(
            r"\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)",
            color=WHITE
        )
        self.place_in_area(chain_rule_gen, "A1", "B6", scale_factor=0.9)
        self.play(Write(chain_rule_gen))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Replace g(x) with y, color y cyan
        self.lecture[1].set_color("#00FFFF")
        chain_rule_y = MathTex(
            r"\frac{d}{dx}[f(", "y", ")] = f'(", "y", ") \cdot \frac{dy}{dx}",
            color=WHITE
        )
        chain_rule_y.set_color_by_tex("y", "#00FFFF")
        self.place_in_area(chain_rule_y, "A1", "B6", scale_factor=0.9)
        
        self.play(Transform(chain_rule_gen, chain_rule_y))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Show d/dx(y^2) below the formula in white
        self.lecture[2].set_color(WHITE)
        deriv_op = MathTex(r"\frac{d}{dx}(", "y", "^2)", color=WHITE)
        deriv_op.set_color_by_tex("y", "#00FFFF")
        self.place_in_area(deriv_op, "C2", "D5", scale_factor=1.1)
        
        self.play(FadeIn(deriv_op, shift=UP * 0.3))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Transform d/dx(y^2) into the result 2y * (dy/dx) using a sliding animation
        self.lecture[3].set_color(WHITE)
        deriv_res = MathTex(r"2", "y", r"\cdot", r"\frac{dy}{dx}", color=WHITE)
        deriv_res.set_color_by_tex("y", "#00FFFF")
        self.place_in_area(deriv_res, "E2", "F5", scale_factor=1.1)
        
        # Sliding transform
        self.play(
            deriv_op.animate.move_to(self.grid["E2"]).set_opacity(0),
            ReplacementTransform(deriv_op.copy(), deriv_res)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # The term dy/dx scales up and turns magenta
        self.lecture[4].set_color("#FF00FF")
        dy_dx_term = deriv_res[3] # Extracting \frac{dy}{dx}
        
        self.play(
            dy_dx_term.animate.scale(1.4).set_color("#FF00FF"),
            run_time=1.5
        )
        self.wait(2)
