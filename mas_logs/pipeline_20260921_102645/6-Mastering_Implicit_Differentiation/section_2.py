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
        self.setup_layout("The 'Secret Sauce': The Chain Rule for y", [
            "Differentiating 'y' requires the Chain Rule.", 
            "Multiply by dy/dx whenever differentiating 'y'.", 
            "Treat 'y' as a hidden function of 'x'."
        ])
        
        # Initial Mobjects
        formula = MathTex(r"\\frac{dy}{dx} = \\frac{dy}{du} \\cdot \\frac{du}{dx}", color=WHITE)
        var_y = MathTex("y", color=BLUE)
        var_x = MathTex("x", color=GREEN)
        var_u = MathTex("u", color=YELLOW)
        bridge = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bridge.svg")

        # === Animation for Lecture Line 1 ===
        # Differentiating 'y' requires the Chain Rule.
        self.lecture[0].set_color("#ADD8E6")
        formula_main = MathTex(r"\\frac{dy}{dx}", color="#ADD8E6")
        self.place_in_area(formula_main, 'B3', 'B5', scale_factor=1.2)
        self.play(Write(formula_main))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Multiply by dy/dx whenever differentiating 'y'.
        self.lecture[1].set_color("#ADD8E6")
        self.place_in_area(formula, 'C3', 'C5', scale_factor=1.0)
        self.play(FadeIn(formula))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Treat 'y' as a hidden function of 'x'.
        self.lecture[2].set_color("#ADD8E6")
        self.place_at_grid(var_y, 'D2', scale_factor=1.0)
        self.place_at_grid(var_u, 'D4', scale_factor=1.0)
        self.place_at_grid(var_x, 'D6', scale_factor=1.0)
        self.place_at_grid(bridge, 'D4', scale_factor=0.3)
        
        arrow1 = Arrow(var_y.get_right(), var_u.get_left())
        arrow2 = Arrow(var_u.get_right(), var_x.get_left())
        
        self.play(FadeIn(var_y), FadeIn(var_u), FadeIn(var_x), FadeIn(bridge), GrowArrow(arrow1), GrowArrow(arrow2))
        self.wait(2)
