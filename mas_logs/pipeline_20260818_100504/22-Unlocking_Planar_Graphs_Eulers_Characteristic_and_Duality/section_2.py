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
        self.setup_layout("The Master Equation: Euler's Characteristic", [
            "Euler's formula: V minus E plus F equals 2.",
            "It holds for any connected planar graph.",
            "Look at this simple triangle example.",
            "Three vertices, three edges, two faces.",
            "Three minus three plus two equals two."
        ])
        
        # Euler Formula: V - E + F = 2
        formula = MathTex("V", "-", "E", "+", "F", "=", "2", font_size=48)
        # Applying requested layout improvement from VideoCritic:
        self.place_at_grid(formula, 'D4', scale_factor=0.85)
        
        # Load asset
        triangle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/triangle.svg")
        self.place_at_grid(triangle, 'B3', scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        # Euler's formula: V - E + F = 2. Color: #FFFFFF. Include triangle asset.
        self.play(FadeIn(formula), FadeIn(triangle))
        self.lecture[0].set_color(WHITE)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight V term in formula. Color: #FF0000.
        self.play(formula[0].animate.set_color("#FF0000"))
        self.lecture[1].set_color("#FF0000")
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        # Highlight E term in formula. Color: #00FF00.
        self.play(formula[2].animate.set_color("#00FF00"))
        self.lecture[2].set_color("#00FF00")
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Highlight F term in formula. Color: #0000FF.
        self.play(formula[4].animate.set_color("#0000FF"))
        self.lecture[3].set_color("#0000FF")
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Highlight result '2'. Color: #FFFF00.
        self.play(formula[6].animate.set_color("#FFFF00"))
        self.lecture[4].set_color("#FFFF00")
        self.wait(2)
