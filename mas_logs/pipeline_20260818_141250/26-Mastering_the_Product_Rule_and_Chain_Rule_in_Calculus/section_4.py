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
        self.setup_layout("Integrated Application: The 'Super-Combo'", [
            "Some functions require both rules.",
            "Identify outer and inner functions first.",
            "Apply rules step-by-step to solve."
        ])
        
        # Define elements
        func = MathTex("f(x) = \\sin(x^2 \\cdot e^x)", font_size=36)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.lecture[0]))
        # Fix for Issue 30: Place in B1-B3
        self.place_in_area(func, "B1", "B3", scale_factor=1.0)
        self.play(Write(func))
        self.lecture[0].set_color("#00FFFF")

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(self.lecture[1]))
        # Identify outer/inner
        outer = MathTex("\\sin(\\dots)", color=RED)
        inner = MathTex("x^2 \\cdot e^x", color=BLUE)
        # Fix for Issue 31: Place closer together
        self.place_at_grid(outer, "D2", scale_factor=0.9)
        self.place_at_grid(inner, "D3", scale_factor=0.9)
        
        self.play(FadeIn(outer), FadeIn(inner))
        self.lecture[1].set_color("#FF00FF")

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(self.lecture[2]))
        # Final expression mockup
        result = MathTex("f'(x) = \\cos(x^2 e^x) \\cdot (2x e^x + x^2 e^x)", color="#FFD700")
        # Fix for Issue 32: Place in E2-E5 and smaller scale
        self.place_in_area(result, "E2", "E5", scale_factor=0.8)
        self.play(Write(result))
        self.lecture[2].set_color("#FFD700")
        
        self.wait(2)
