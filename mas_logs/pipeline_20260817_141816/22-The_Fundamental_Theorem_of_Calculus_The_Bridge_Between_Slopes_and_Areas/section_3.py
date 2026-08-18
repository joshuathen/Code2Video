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
        lecture_lines = [
            "The derivative of accumulation returns the original function.",
            "As the width expands, area increases instantly.",
            "This rate of change equals the function's height.",
            "Thus, F prime of x equals f of x.",
            "Area growth rate mirrors the function's current value."
        ]
        self.setup_layout("The Fundamental Theorem (Part 1)", lecture_lines)
        
        # Assets
        calc_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg")
        graph_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/graph.svg")
        
        # Mobjects
        theorem_text = Text("Fundamental Theorem", color=WHITE)
        integral_expr = MathTex(r"\\int_a^x f(t) dt", color="#00CCFF")
        derivative_expr = MathTex(r"\\frac{d}{dx} F(x) = f(x)", color="#FF9900")

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.place_at_grid(theorem_text, 'B4', scale_factor=0.6)
        self.place_at_grid(calc_icon, 'B2', scale_factor=0.4)
        self.play(FadeIn(theorem_text), FadeIn(calc_icon))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00CCFF"))
        self.place_in_area(integral_expr, 'C3', 'C5', scale_factor=0.75)
        self.play(Write(integral_expr))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF9900"))
        self.place_in_area(derivative_expr, 'D3', 'D5', scale_factor=0.75)
        self.play(Write(derivative_expr))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FFFFFF"))
        self.play(Indicate(derivative_expr))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FF9900"))
        self.place_at_grid(graph_icon, 'E4', scale_factor=0.4)
        self.play(FadeIn(graph_icon), Circumscribe(derivative_expr))
        self.wait(2)
