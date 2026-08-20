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
        self.setup_layout("The True Mathematical Pattern", [
            "Powers of two fail here.",
            "Use Euler's characteristic instead.",
            "The formula is a sum.",
            "Regions equal C(n,4) plus C(n,2) plus one.",
            "This works for all n."
        ])
        
        # Assets
        calculator = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg")
        
        # === Animation for Lecture Line 1 ===
        # Show title in #FFFFFF with icon
        self.place_at_grid(calculator, 'A6', scale_factor=0.3)
        self.play(FadeIn(calculator))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Display the correct formula in #FF5733
        formula = MathTex("R = \\binom{n}{4} + \\binom{n}{2} + 1", color="#FF5733")
        self.place_in_area(formula, 'B2', 'C5', scale_factor=1.0)
        self.play(Write(formula))
        self.lecture[1].set_color("#FF5733")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Animate substitution of n=6 into formula in #33FF57
        sub_formula = MathTex("R = \\binom{6}{4} + \\binom{6}{2} + 1", color="#33FF57")
        self.place_in_area(sub_formula, 'D2', 'D5', scale_factor=0.9)
        self.play(ReplacementTransform(formula.copy(), sub_formula))
        self.lecture[2].set_color("#33FF57")
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Show result 31 in #FFFF33
        result = MathTex("R = 15 + 15 + 1 = 31", color="#FFFF33")
        self.place_at_grid(result, 'E2', scale_factor=1.0)
        self.play(Write(result))
        self.lecture[3].set_color("#FFFF33")
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Show final result 31 in #33FF57 with calculator icon
        self.play(FadeOut(formula), FadeOut(sub_formula), FadeOut(result))
        final_result = MathTex("31", color="#33FF57", font_size=72)
        self.place_at_grid(final_result, 'E4', scale_factor=1.5)
        
        # Update calculator position
        self.play(calculator.animate.move_to(self.grid['C4']), Write(final_result))
        self.lecture[4].set_color("#FFFFFF")
        self.wait(2)
