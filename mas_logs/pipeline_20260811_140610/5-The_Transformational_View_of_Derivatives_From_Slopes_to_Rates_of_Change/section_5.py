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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Derivative is the instantaneous rate of change.", 
                         "It represents the tangent line's slope.", 
                         "We observe how things are changing."]
        self.setup_layout("Summary and Synthesis", lecture_lines)
        
        slope_color = "#E74C3C"
        derivative_color = "#9B59B6"
        
        # Elements
        concept_slope = Text("Slope", color=slope_color)
        concept_secant = Text("Secant", color=WHITE)
        concept_tangent = Text("Tangent", color=WHITE)
        concept_derivative = Text("Derivative", color=derivative_color)
        
        formula = MathTex(
            r"f'(x) = \lim_{h \to 0} \frac{f(x+h)-f(x)}{h}",
            font_size=32
        )
        
        # Animation
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(derivative_color))
        self.place_at_grid(concept_derivative, 'B2', scale_factor=0.8)
        self.place_at_grid(concept_slope, 'B5', scale_factor=0.8)
        self.play(FadeIn(concept_derivative), FadeIn(concept_slope))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(slope_color))
        self.place_at_grid(concept_tangent, 'C2', scale_factor=0.8)
        self.place_at_grid(concept_secant, 'C5', scale_factor=0.8)
        self.play(FadeIn(concept_tangent), FadeIn(concept_secant))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.place_in_area(formula, 'E2', 'F5', scale_factor=0.9)
        self.play(Write(formula))
        self.wait(2)
