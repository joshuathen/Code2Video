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
            "This is the formal definition of a derivative.",
            "It is the limit of the difference quotient.",
            "The slope at a single point.",
            "It captures the instantaneous rate of change.",
            "A precise measurement at one moment."
        ]
        self.setup_layout("Formal Definition: The Derivative", lecture_lines)
        
        # Pre-build objects
        limit_expr = MathTex(r"f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}", font_size=40)
        speedometer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/speedometer.svg")
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.lecture[0]))
        # Applying requested layout fix (Issue 26 / 37)
        self.place_in_area(limit_expr, 'B3', 'D5', scale_factor=1.2)
        self.play(Write(limit_expr))

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(self.lecture[1]))
        self.lecture[0].set_color(GRAY)
        self.lecture[1].set_color(BLUE)

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(self.lecture[2]))
        self.lecture[1].set_color(GRAY)
        self.lecture[2].set_color(YELLOW)
        limit_expr.set_color_by_tex(r"f'(x)", YELLOW)
        # Placeholder for limit_definition_anim (Issue 27 / 37)
        # Using a rectangle as a stand-in for the requested animation asset
        limit_definition_anim = Rectangle(width=1.5, height=1.0, color=BLUE)
        self.place_at_grid(limit_definition_anim, 'E2', scale_factor=0.6)
        self.play(FadeIn(limit_definition_anim))

        # === Animation for Lecture Line 4 ===
        self.play(FadeIn(self.lecture[3]))
        self.lecture[2].set_color(GRAY)
        self.lecture[3].set_color(GREEN)

        # === Animation for Lecture Line 5 ===
        self.play(FadeIn(self.lecture[4]))
        self.lecture[3].set_color(GRAY)
        self.lecture[4].set_color(ORANGE)
        # Anchoring speedometer (Issue 28 / 37)
        self.place_at_grid(speedometer, 'E5', scale_factor=0.6)
        self.play(FadeIn(speedometer))
        self.wait(1)
