from manim import *
import numpy as np

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT, buff=0.4).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.5)
        self.add(self.lecture)

        # Define fine-grained animation grid (6x6 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                # Offset grid to the right side of the screen
                x = 1.0 + j * 1.0
                y = 2.2 - i * 1.0
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
        # Define lecture content
        lecture_content = [
            "- Rule: [f(g(x)) * h(x)]'",
            "- First: Apply Product Rule",
            "- Second: Apply Chain Rule",
            "- Inside: (u v)' = u'v + uv'",
            "- Outer: f'(g(x)) * g'(x)"
        ]
        
        self.setup_layout("Mastering Product and Chain Rules", lecture_content)

        # Problem Statement
        problem = MathTex(r"y = x^2 \cdot \sin(3x^4)", color=BLUE)
        self.place_at_grid(problem, "A3", scale_factor=0.9)
        
        # Step 1: Product Rule Identification
        step1 = MathTex(r"u = x^2, \quad v = \sin(3x^4)", font_size=32)
        self.place_at_grid(step1, "B3", scale_factor=0.8)
        
        # Step 2: Derivatives
        step2_u = MathTex(r"u' = 2x", font_size=32)
        step2_v = MathTex(r"v' = \cos(3x^4) \cdot 12x^3", font_size=32)
        self.place_at_grid(step2_u, "C2", scale_factor=0.8)
        self.place_at_grid(step2_v, "C4", scale_factor=0.8)
        
        # Step 3: Combine
        step3 = MathTex(
            r"y' = (2x)\sin(3x^4) + (x^2)(12x^3 \cos(3x^4))", 
            font_size=32
        )
        self.place_at_grid(step3, "D3", scale_factor=0.8)
        
        # Final Result
        final = MathTex(
            r"y' = 2x\sin(3x^4) + 12x^5\cos(3x^4)", 
            color=YELLOW, 
            font_size=34
        )
        self.place_at_grid(final, "E3", scale_factor=0.9)

        # Animations
        self.play(Write(problem))
        self.wait(1)
        self.play(FadeIn(step1))
        self.wait(0.5)
        self.play(Write(step2_u), Write(step2_v))
        self.wait(1)
        self.play(TransformMatchingShapes(step1.copy(), step3))
        self.wait(1)
        self.play(Write(final))
        self.play(Indicate(final))
        self.wait(2)

if __name__ == "__main__":
    with tempconfig({"quality": "medium_quality", "preview": True}):
        scene = Section3Scene()
        scene.render()