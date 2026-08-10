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
        self.setup_layout("Formalizing the Derivative Definition", [
            "Limit notation defines the exact rate.",
            "F-prime of x equals the limit.",
            "The distance h shrinks toward zero.",
            "This gives the instantaneous speed here.",
            "Like a speedometer reading right now."
        ])
        
        # Define equations
        diff_quotient = MathTex(r"\frac{f(x+h) - f(x)}{h}", color=WHITE)
        limit_expr = MathTex(r"\lim_{h \to 0}", color=WHITE)
        derivative = MathTex(r"f'(x) = ", r"\lim_{h \to 0} \frac{f(x+h) - f(x)}{h}", color=WHITE)
        derivative[1].set_color("#32CD32")
        
        speed_label = Text("Speed at instant", color="#32CD32", font_size=24)
        speedometer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/speedometer.svg").scale(0.5)

        # === Animation for Lecture Line 1 ===
        self.place_at_grid(diff_quotient, 'B4', scale_factor=0.8)
        self.play(FadeIn(diff_quotient))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        self.place_at_grid(limit_expr, 'B2', scale_factor=0.7)
        self.play(FadeIn(limit_expr))
        self.lecture[1].set_color(YELLOW)

        # === Animation for Lecture Line 3 ===
        self.play(Indicate(limit_expr))
        self.lecture[2].set_color(YELLOW)

        # === Animation for Lecture Line 4 ===
        # Transform to formal derivative notation
        self.place_in_area(derivative, 'D2', 'F5', scale_factor=0.9)
        self.play(FadeOut(limit_expr), FadeOut(diff_quotient), FadeIn(derivative))
        self.lecture[3].set_color(YELLOW)

        # === Animation for Lecture Line 5 ===
        # Highlight derivative symbol and show assets
        speed_group = VGroup(speed_label, speedometer).arrange(DOWN)
        self.place_at_grid(speed_group, 'E3', scale_factor=1.0)
        self.play(Write(speed_label), FadeIn(speedometer))
        self.play(Indicate(derivative[0]))
        self.lecture[4].set_color(YELLOW)
        self.wait(2)
