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
        lecture_lines = [
            "Cramer's Rule is a ratio of areas.",
            "x equals the area ratio.",
            "Geometric scaling determines the solution."
        ]
        self.setup_layout("Synthesis: The Ratio Formula", lecture_lines)
        
        # Formula: x_1 = det(A_1) / det(A)
        formula = MathTex(
            r"x_1 = \frac{\det(A_1)}{\det(A)}",
            font_size=40
        )
        self.place_in_area(formula, 'B3', 'B4', scale_factor=0.9)

        # Asset placeholders (none.svg does not exist, using SVG-like shape if needed or simply ignoring if "none")
        # Since it is a placeholder, creating a simple dummy
        icon = Circle(radius=0.2, color=WHITE) 

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(formula), FadeIn(icon.copy()))
        self.lecture[0].set_color("#FFD700") # Gold

        # === Animation for Lecture Line 2 ===
        # Represent area ratio
        area_a = Square(side_length=2, color=BLUE).set_opacity(0.3)
        area_a1 = Rectangle(width=2, height=4, color=RED).set_opacity(0.3)
        
        self.place_at_grid(area_a, 'D2', scale_factor=0.7)
        self.place_at_grid(area_a1, 'D5', scale_factor=0.7)
        
        self.play(FadeIn(area_a), FadeIn(area_a1))
        self.lecture[1].set_color("#00FFFF") # Cyan

        # === Animation for Lecture Line 3 ===
        self.play(formula.animate.set_color("#00FF00"), FadeIn(icon))
        self.lecture[2].set_color("#00FF00") # Lime
        self.wait(2)
