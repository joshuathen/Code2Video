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
            "If X and Y are Normal, their sum is Normal.",
            "The new mean is the sum of means.",
            "The new variance is the sum of variances.",
            "Note: Variances add, not standard deviations.",
            "This simple rule handles complex error accumulation."
        ]
        self.setup_layout("The Core Rule: Adding Means and Variances", lecture_lines)
        
        # Define equations
        eq1 = MathTex("X \\sim N(\\mu_1, \\sigma_1^2), \\; Y \\sim N(\\mu_2, \\sigma_2^2)", font_size=30)
        eq2 = MathTex("X + Y \\sim N(\\mu_1 + \\mu_2, \\sigma_1^2 + \\sigma_2^2)", font_size=32, color=YELLOW)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.place_at_grid(eq1, "C3", scale_factor=0.9)
        self.play(Write(eq1))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(GREEN)
        self.play(Indicate(eq1))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        self.place_at_grid(eq2, "F3", scale_factor=0.9)
        self.play(ReplacementTransform(eq1.copy(), eq2))
        self.play(Write(eq2))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(RED)
        warning = Text("! Do not add standard deviations !", font_size=24, color=RED)
        self.place_at_grid(warning, "D2", scale_factor=0.8)
        self.play(FadeIn(warning))
        self.play(Indicate(warning))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(PURPLE)
        self.play(FadeOut(warning), FadeOut(eq1))
        self.play(eq2.animate.move_to(self.grid["C3"]))
