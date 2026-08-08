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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Unified Framework (Prerequisite Review)", [
            "Repeated multiplication is our base growth.",
            "A seed grows into a result.",
            "Base to the exponent equals result.",
            "Three squared equals nine.",
            "Growth follows this powerful structure."
        ])
        
        # Math objects
        eq_exp = MathTex("x = b^{y}", font_size=48, color=WHITE)
        eq_log = MathTex("y = \\log_{b}(x)", font_size=48, color=WHITE)
        seed_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/seed.svg", color=GREEN)
        
        # === Animation for Lecture Line 1 ===
        self.place_in_area(eq_exp, 'A2', 'B5', scale_factor=0.6)
        self.play(FadeIn(eq_exp))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        exponent_y = eq_exp[0][-1]
        self.place_at_grid(seed_icon, 'B5', scale_factor=0.3)
        self.play(FadeIn(seed_icon))
        glow = Circle(radius=0.3, color=YELLOW, stroke_width=2).move_to(exponent_y)
        self.play(Create(glow), run_time=1.5)
        self.play(FadeOut(glow))
        self.lecture[1].set_color(YELLOW)

        # === Animation for Lecture Line 3 ===
        self.place_in_area(eq_log, 'D2', 'E5', scale_factor=0.6)
        self.play(FadeIn(eq_log))
        self.lecture[2].set_color(YELLOW)

        # === Animation for Lecture Line 4 ===
        arrow = Arrow(eq_exp.get_bottom(), eq_log.get_top(), color=BLUE)
        self.place_at_grid(arrow, 'C3', scale_factor=0.5)
        self.play(GrowArrow(arrow))
        self.lecture[3].set_color(YELLOW)

        # === Animation for Lecture Line 5 ===
        seed_icon.move_to(self.grid['F3'])
        self.play(Indicate(eq_exp, color=GREEN), Indicate(eq_log, color=GREEN), Indicate(seed_icon, color=GREEN))
        self.lecture[4].set_color(YELLOW)
        self.wait(2)
