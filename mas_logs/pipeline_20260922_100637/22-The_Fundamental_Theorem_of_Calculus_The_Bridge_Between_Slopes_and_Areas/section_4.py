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
        self.setup_layout("The Fundamental Theorem (Part 2) & Application", [
            "We evaluate definite integrals easily.",
            "Use the antiderivative at bounds.",
            "Calculate F of b minus F of a."
        ])
        
        # Load asset
        calculator = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg")
        
        # Setup Math expressions
        formula = MathTex(r"\int_{a}^{b} f(x) \, dx = F(b) - F(a)", font_size=36)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.title))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF5733"))
        # Fix for Issue 28 & 30: Use place_in_area as requested
        self.place_in_area(formula, 'B3', 'B6', scale_factor=0.9)
        self.play(Write(formula))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#33FF57"))
        final_text = MathTex(r"= 9", color="#33FF57", font_size=36)
        # Fix for Issue 29
        self.place_at_grid(final_text, 'C4', scale_factor=1.2)
        # Add asset
        self.place_at_grid(calculator, 'D4', scale_factor=0.5)
        
        self.play(Write(final_text), FadeIn(calculator))
        self.wait(2)
