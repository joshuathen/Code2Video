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
        self.setup_layout("Synthesis & Complex Application", [
            "Combine rules for complex problems.",
            "Identify product or chain rule first.",
            "Break it down into logical steps."
        ])

        # Assets
        blueprint = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/blueprint.svg")
        puzzle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/puzzle.svg")

        # Complex Expression
        expression = MathTex("h(x) = (x^2 + 1)^3 \\cdot \\cos(x)", font_size=42)
        # Fix 30: Adjust position
        self.place_in_area(expression, "B4", "B6", scale_factor=0.9)
        self.place_at_grid(blueprint, "A5", scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#3498DB"))
        self.play(Write(expression), FadeIn(blueprint))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#E74C3C"))
        # Fix 31: Adjust positions
        part1 = MathTex("(x^2 + 1)^3", color="#3498DB")
        part2 = MathTex("\\cos(x)", color="#E74C3C")
        self.place_at_grid(part1, "D3", scale_factor=0.8)
        self.place_at_grid(part2, "D5", scale_factor=0.8)
        self.play(FadeIn(part1), FadeIn(part2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#2ECC71"))
        # Fix 32: Adjust position
        result = MathTex("h'(x) = 3(x^2+1)^2(2x)\\cos(x) - (x^2+1)^3\\sin(x)", color="#2ECC71", font_size=32)
        self.place_at_grid(result, "E4", scale_factor=1.0)
        self.place_at_grid(puzzle, "F6", scale_factor=0.5)
        self.play(Write(result), FadeIn(puzzle))
        self.wait(2)
