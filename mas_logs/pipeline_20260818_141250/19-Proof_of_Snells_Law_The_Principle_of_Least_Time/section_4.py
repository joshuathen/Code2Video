from manim import *
import numpy as np

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
        self.setup_layout("Derivation: The Calculus of Refraction", [
            "Differentiate time with respect to distance.",
            "Set the derivative to zero.",
            "Relate geometry to sine functions.",
            "Equate sine and velocity ratios.",
            "Geometric proof of refraction law."
        ])

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        t_eq = MathTex("T(x) = \\frac{\sqrt{a^2+x^2}}{v_1} + \\frac{\sqrt{b^2+(L-x)^2}}{v_2}", font_size=32)
        self.place_at_grid(t_eq, "B2")
        self.play(Write(t_eq))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFD700")
        diff_op = MathTex("\\frac{dT}{dx} = \\frac{x}{v_1\\sqrt{a^2+x^2}} - \\frac{L-x}{v_2\\sqrt{b^2+(L-x)^2}} = 0", font_size=32)
        self.place_at_grid(diff_op, "C2")
        self.play(Write(diff_op))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(WHITE)
        geom_rel = MathTex("\\sin \\theta_1 = \\frac{x}{\\sqrt{a^2+x^2}}, \\quad \\sin \\theta_2 = \\frac{L-x}{\\sqrt{b^2+(L-x)^2}}", font_size=32)
        self.place_at_grid(geom_rel, "D2")
        self.play(FadeIn(geom_rel))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#00FF00")
        sub_step = MathTex("\\frac{\\sin \\theta_1}{v_1} - \\frac{\\sin \\theta_2}{v_2} = 0", font_size=36)
        self.place_at_grid(sub_step, "E2")
        self.play(ReplacementTransform(VGroup(diff_op, geom_rel), sub_step))
        sub_step.set_color_by_tex("\\sin \\theta_1", BLUE)
        sub_step.set_color_by_tex("\\sin \\theta_2", BLUE)
        sub_step.set_color_by_tex("v_1", GREEN)
        sub_step.set_color_by_tex("v_2", GREEN)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(WHITE)
        final_eq = MathTex("\\frac{\\sin \\theta_1}{v_1} = \\frac{\\sin \\theta_2}{v_2}", font_size=42, color=YELLOW)
        self.place_at_grid(final_eq, "E5")
        self.play(Transform(sub_step, final_eq))
