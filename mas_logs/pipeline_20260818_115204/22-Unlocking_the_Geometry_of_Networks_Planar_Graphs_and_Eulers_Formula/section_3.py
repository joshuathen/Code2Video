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
            "Euler’s formula relates V, E, and F.",
            "V minus E plus F equals 2.",
            "This constant holds for planar graphs.",
            "Check it with a simple square.",
            "The formula remains true regardless of complexity."
        ]
        self.setup_layout("Euler’s Characteristic Formula", lecture_lines)
        
        # Pre-create elements
        formula = MathTex("V", "-", "E", "+", "F", "=", "2", font_size=48)
        v_part = formula[0] # V
        e_part = formula[2] # E
        f_part = formula[4] # F
        
        # Load asset
        square_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/square.svg")
        
        # 62: Use place_in_area as requested
        self.place_in_area(formula, 'B3', 'C5', scale_factor=0.9)
        self.place_at_grid(square_asset, 'C4', scale_factor=0.8) # Place behind/around formula
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(formula), FadeIn(square_asset), self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            Indicate(v_part, color="#00FF00"),
            Indicate(e_part, color="#0000FF"),
            Indicate(f_part, color="#FF0000"),
            self.lecture[1].animate.set_color("#FFFFFF")
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(formula.animate.set_color("#FFFF00"), self.lecture[2].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Simple square example: V=4, E=5, F=3
        graph = VGroup(
            Square(side_length=1.5),
            Line(np.array([-0.75, 0.75, 0]), np.array([0.75, -0.75, 0]))
        )
        # 87: Use place_at_grid with E5 as requested
        self.place_at_grid(graph, 'E5', scale_factor=0.6)
        self.play(Create(graph), self.lecture[3].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            FadeOut(graph),
            FadeOut(square_asset),
            formula.animate.move_to(ORIGIN).scale(1.5),
            self.lecture[4].animate.set_color("#FFFFFF")
        )
        self.wait(2)
