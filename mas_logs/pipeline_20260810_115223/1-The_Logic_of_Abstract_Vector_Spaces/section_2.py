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
        lecture_lines = [
            "A vector space is a collection of objects.",
            "It satisfies eight fundamental axioms.",
            "Vectors aren't just physical arrows.",
            "Functions can be vectors.",
            "Polynomials can be vectors too."
        ]
        self.setup_layout("The Generalization Leap", lecture_lines)
        
        # Assets (Using placeholders where real files aren't found)
        # Using SVG for Arrow from /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg
        # Using MathTex/Text as requested
        try:
            arrow = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg", color="#FF5733")
        except:
            arrow = Arrow(start=ORIGIN, end=RIGHT, color="#FF5733")
        
        func_text = MathTex("f(x)", color="#33FF57")
        poly_text = MathTex("ax^2 + bx + c", color="#3357FF")
        
        # Initial placement
        self.place_at_grid(arrow, 'D3', scale_factor=1.0)
        self.add(arrow)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF5733")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(WHITE)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF5733")
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#33FF57")
        self.play(Transform(arrow, self.place_in_area(func_text, 'B3', 'B4', scale_factor=1.2)))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#3357FF")
        self.play(Transform(arrow, self.place_in_area(poly_text, 'E3', 'E4', scale_factor=1.2)))
        self.wait(1)
