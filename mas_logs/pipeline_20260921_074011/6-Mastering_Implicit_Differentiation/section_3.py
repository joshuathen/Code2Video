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
        self.setup_layout("The Step-by-Step Algorithm", [
            "Step 1: Differentiate both sides regarding x.",
            "Step 2: Collect all dy/dx terms together.",
            "Step 3: Factor out and solve for dy/dx."
        ])
        
        equation = MathTex("x^3 + y^3 = 6xy", font_size=36)
        self.place_in_area(equation, 'A2', 'B5', scale_factor=0.9)
        self.add(equation)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        step1_eq = MathTex(r"\frac{d}{dx}(x^3 + y^3) = \frac{d}{dx}(6xy)", font_size=32, color=WHITE)
        self.place_at_grid(step1_eq, "C3")
        self.play(Write(step1_eq))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(YELLOW))
        
        step2_eq = MathTex(r"3x^2 + 3y^2 \frac{dy}{dx} = 6y + 6x \frac{dy}{dx}", font_size=32)
        step2_grouped = MathTex(r"(3y^2 - 6x) \frac{dy}{dx} = 6y - 3x^2", font_size=32, color="#FFD700")
        self.place_in_area(step2_eq, 'C2', 'D5', scale_factor=0.85)
        self.play(Write(step2_eq))
        self.wait(1)
        self.place_in_area(step2_grouped, 'E2', 'F5', scale_factor=0.85)
        self.play(ReplacementTransform(step2_eq, step2_grouped))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(YELLOW))
        
        result = MathTex(r"\frac{dy}{dx} = \frac{6y - 3x^2}{3y^2 - 6x}", font_size=36, color="#FF4500")
        self.place_at_grid(result, 'E5', scale_factor=0.9)
        self.play(Write(result))
        self.wait(2)
