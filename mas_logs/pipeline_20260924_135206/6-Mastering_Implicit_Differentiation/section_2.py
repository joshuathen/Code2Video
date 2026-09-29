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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Core Concept: The Chain Rule", [
            "Recall the chain rule for derivatives.",
            "Treat y as an inner function, y(x).",
            "Differentiating y gives dy/dx term."
        ])
        
        # Define the equation
        eq = MathTex(r"{dy \over dx} = {dy \over du} \cdot {du \over dx}", font_size=40)
        # Apply fix for issues 27/29/42/44
        self.place_in_area(eq, 'B2', 'C5', scale_factor=0.9)
        
        # === Animation for Lecture Line 1 ===
        self.play(Write(eq))
        self.play(self.lecture[0].animate.set_color('#FFD700'))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight terms
        self.play(self.lecture[1].animate.set_color('#00CED1'))
        self.play(Indicate(eq))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color('#FF4500'))
        # Represent small increment (dy, dx)
        dot = Dot(color='#FF4500')
        label_u = Text('Variable u', font_size=20)
        # Apply fix for issues 28/43
        self.place_at_grid(dot, 'E3', scale_factor=0.5)
        self.place_at_grid(label_u, 'E4', scale_factor=0.6)
        
        self.play(FadeIn(dot), Write(label_u))
        arrow_dx = Arrow(start=dot.get_center(), end=dot.get_center() + RIGHT*0.5, color=WHITE)
        arrow_dy = Arrow(start=dot.get_center(), end=dot.get_center() + UP*0.5, color=WHITE)
        self.play(Create(arrow_dx), Create(arrow_dy))
        self.wait(2)
