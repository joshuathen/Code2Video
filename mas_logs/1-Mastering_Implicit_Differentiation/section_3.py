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
        # Mandatory setup of the layout
        lecture_lines = [
            "First, differentiate both sides of the equation.",
            "Apply the chain rule to every term containing y.",
            "Group all terms with dy dx on one side.",
            "Factor out dy dx and isolate it.",
            "The result is our formula for the slope."
        ]
        self.setup_layout("The Four-Step Process", lecture_lines)

        # Define custom colors
        COLOR_X = "#00AAFF"    # blue
        COLOR_Y = "#00FF00"    # green
        COLOR_DYDX = "#FF0000" # red
        COLOR_HIGHLIGHT = "#FFFF00" # yellow

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_HIGHLIGHT))
        
        eq1 = VGroup(Text("x^2"), Text("+"), Text("y^2"), Text("="), Text("25")).arrange(RIGHT, buff=0.2)
        eq1[0].set_color(COLOR_X)
        eq1[2].set_color(COLOR_Y)
        self.place_at_grid(eq1, "B3", scale_factor=1.0)
        self.play(Write(eq1))
        self.wait(1)

        eq1_diff = VGroup(Text("d/dx"), Text("("), Text("x^2"), Text("+"), Text("y^2"), Text(")"), Text("="), Text("d/dx"), Text("("), Text("25"), Text(")")).arrange(RIGHT, buff=0.2)
        eq1_diff[2].set_color(COLOR_X)
        eq1_diff[4].set_color(COLOR_Y)
        self.place_in_area(eq1_diff, "B2", "B5", scale_factor=1.0)
        self.play(ReplacementTransform(eq1, eq1_diff))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_HIGHLIGHT)
        )

        # Fixed by using VGroup and Text to avoid LaTeX dependency
        eq2 = VGroup(Text("2"), Text("x"), Text("+"), Text("2"), Text("y"), Text("dy/dx"), Text("="), Text("0")).arrange(RIGHT, buff=0.1)
        eq2[1].set_color(COLOR_X)
        eq2[4].set_color(COLOR_Y)
        eq2[5].set_color(COLOR_DYDX)
        self.place_at_grid(eq2, "C3", scale_factor=1.0)
        self.play(ReplacementTransform(eq1_diff, eq2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_HIGHLIGHT)
        )

        # Fixed by using VGroup and Text to avoid LaTeX dependency
        eq3 = VGroup(Text("2"), Text("y"), Text("dy/dx"), Text("="), Text("-2"), Text("x")).arrange(RIGHT, buff=0.1)
        eq3[1].set_color(COLOR_Y)
        eq3[2].set_color(COLOR_DYDX)
        eq3[5].set_color(COLOR_X)
        self.place_at_grid(eq3, "D3", scale_factor=1.0)
        self.play(ReplacementTransform(eq2, eq3))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(COLOR_HIGHLIGHT)
        )

        # Fixed by using VGroup and Text to avoid LaTeX dependency
        eq4 = VGroup(Text("dy/dx"), Text("="), Text("-2"), Text("x"), Text("/ 2"), Text("y"), Text("")).arrange(RIGHT, buff=0.1)
        eq4[0].set_color(COLOR_DYDX)
        eq4[3].set_color(COLOR_X)
        eq4[5].set_color(COLOR_Y)
        self.place_at_grid(eq4, "E3", scale_factor=1.0)
        self.play(ReplacementTransform(eq3, eq4))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(COLOR_HIGHLIGHT)
        )

        # Fixed by using VGroup and Text to avoid LaTeX dependency
        eq5 = VGroup(Text("dy/dx"), Text("="), Text("-"), Text("x"), Text("/"), Text("y"), Text("")).arrange(RIGHT, buff=0.1)
        eq5[0].set_color(COLOR_DYDX)
        eq5[3].set_color(COLOR_X)
        eq5[5].set_color(COLOR_Y)
        self.place_at_grid(eq5, "F3", scale_factor=1.0)
        self.play(ReplacementTransform(eq4, eq5))
        self.wait(2)
