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
        self.setup_layout("Defining Independence: Does B change A?", [
            "Events are independent if B doesn't change A.",
            "Mathematically, P(A|B) must equal P(A).",
            "Knowing B gives zero new information about A."
        ])
        
        # Elements
        circle_a = Circle(radius=1.2, color=BLUE, fill_opacity=0.3)
        circle_b = Circle(radius=1.2, color=YELLOW, fill_opacity=0.3)
        circle_b.shift(RIGHT * 1.2)
        sets = VGroup(circle_a, circle_b)
        
        intersection = Intersection(circle_a, circle_b, color="#FF00FF", fill_opacity=0.6)
        formula = MathTex(r"P(A|B) = \frac{P(A \cap B)}{P(B)}", font_size=36, color="#00FF00")

        # === Animation for Lecture Line 1 ===
        self.place_in_area(sets, "B2", "E5", scale_factor=0.6)
        self.play(FadeIn(sets))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(intersection))
        self.lecture[1].set_color("#FF00FF")

        # === Animation for Lecture Line 3 ===
        self.place_at_grid(formula, "C3", scale_factor=1.0)
        self.play(Write(formula))
        self.play(Flash(formula, color="#00FF00"))
        self.lecture[2].set_color("#00FF00")
        
        self.wait(2)
