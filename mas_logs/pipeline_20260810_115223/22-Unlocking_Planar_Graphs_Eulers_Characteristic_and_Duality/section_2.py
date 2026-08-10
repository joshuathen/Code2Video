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
        self.setup_layout("The Golden Rule: Euler's Characteristic Formula", [
            "Euler's formula defines a fundamental invariant.",
            "Sum Vertices, subtract Edges, add Faces.",
            "For connected graphs, this equals two.",
            "It holds regardless of graph complexity.",
            "The value two always remains constant."
        ])
        
        formula = MathTex("V - E + F = 2", font_size=48, color=WHITE)
        triangle_calc = MathTex("3 - 3 + 2 = 2", font_size=48, color="#00FFFF")
        complex_calc = MathTex("4 - 6 + 4 = 2", font_size=48, color="#00FF00")
        
        triangle_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/triangle.svg")
        square_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/square.svg")

        # === Animation for Lecture Line 1 ===
        self.place_in_area(formula, 'B2', 'C5', scale_factor=0.9)
        self.play(Write(formula))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        self.place_at_grid(triangle_img, 'D2', scale_factor=0.6)
        self.play(FadeIn(triangle_img))
        self.lecture[1].set_color("#00FFFF")

        # === Animation for Lecture Line 3 ===
        self.place_in_area(triangle_calc, 'D3', 'D5', scale_factor=0.85)
        self.play(Write(triangle_calc))
        self.lecture[2].set_color("#00FFFF")

        # === Animation for Lecture Line 4 ===
        self.play(FadeOut(triangle_img), FadeOut(triangle_calc))
        self.place_at_grid(square_img, 'D2', scale_factor=0.6)
        self.play(FadeIn(square_img))
        self.lecture[3].set_color("#FFFFFF")

        # === Animation for Lecture Line 5 ===
        self.place_in_area(complex_calc, 'D3', 'D5', scale_factor=0.85)
        self.play(Write(complex_calc))
        self.lecture[4].set_color("#00FF00")
        self.wait(2)
