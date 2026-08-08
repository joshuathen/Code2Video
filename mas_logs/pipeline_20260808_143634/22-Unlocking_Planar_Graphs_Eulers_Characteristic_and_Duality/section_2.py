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
        self.setup_layout("Euler’s Characteristic Formula", [
            "Euler's formula: V - E + F = 2.",
            "Connected planar graphs follow this rule.",
            "Face F includes the infinite outer region.",
            "Example: Triangle has 3V, 3E, 2F.",
            "Calculation confirms 3 - 3 + 2 = 2."
        ])
        
        # Load asset
        triangle_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/triangle.svg")
        
        formula = MathTex("V", "-", "E", "+", "F", "=", "2")
        formula.set_color_by_tex("V", RED)
        formula.set_color_by_tex("E", GREEN)
        formula.set_color_by_tex("F", BLUE)
        self.place_at_grid(formula, 'A4', scale_factor=0.6)
        
        # Position triangle
        self.place_in_area(triangle_icon, 'C4', 'F6', scale_factor=0.8)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFF00")
        self.play(Write(formula), FadeIn(triangle_icon))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFF00")
        self.play(FadeIn(triangle_icon))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00")
        self.play(Flash(formula[4], color=BLUE))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFFF00")
        self.play(Flash(formula[0], color=RED), Flash(formula[2], color=GREEN), Flash(formula[4], color=BLUE))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFFF00")
        circle = SurroundingRectangle(formula, color=YELLOW)
        self.play(Create(circle))
        self.wait(2)
