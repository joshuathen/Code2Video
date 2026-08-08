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
            "We define the slope as a limit.",
            "The formula uses a difference quotient.",
            "As h approaches zero, we get the derivative.",
            "It maps the slope at any input x.",
            "The derivative gives the instantaneous rate."
        ]
        self.setup_layout("Defining the Derivative", lecture_lines)
        
        # Formula setup
        formula = MathTex(
            "f'(x) = \\lim_{h \\to 0} \\frac{f(x+h) - f(x)}{h}",
            font_size=40
        )
        
        # Asset Loading
        pen = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pen.svg")
        graph_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/graph.svg")
        
        # Define colored parts
        formula.set_color_by_tex("f(x+h)", "#FF0000")
        formula.set_color_by_tex("f(x)", "#00FF00")
        
        # Initial placement
        self.place_in_area(formula, 'B3', 'D6', scale_factor=0.7)
        self.place_at_grid(pen, 'A6', scale_factor=0.3)
        self.place_at_grid(graph_icon, 'F6', scale_factor=0.3)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(formula), FadeIn(pen), self.lecture[0].animate.set_color(WHITE))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(WHITE))
        # Animate division by h (Yellow) - Wrapped in Wait/ReplacementTransform to avoid calling method in play
        self.play(formula.animate.set_color_by_tex("h", "#FFFF00"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(WHITE))
        # Pulse limit notation
        self.play(Indicate(formula.get_parts_by_tex("\\lim_{h \\to 0}"), color="#00FFFF"))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(WHITE))
        # Highlight f'(x)
        self.play(Indicate(formula.get_parts_by_tex("f'(x)"), color="#FF00FF"))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(WHITE), FadeIn(graph_icon))
        self.play(formula.animate.set_color("#FF00FF"))
        self.wait(2)
