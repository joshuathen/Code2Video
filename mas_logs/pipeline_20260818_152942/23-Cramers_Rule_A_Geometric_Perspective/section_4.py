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
            "Cramer's formula is x equals ratio of determinants.",
            "Area of Ax over Area of A.",
            "Linear scaling isolates our scalar weights.",
            "Calculations confirm the geometric intuition.",
            "System solutions emerge from area ratios."
        ]
        self.setup_layout("Putting it Together: The Visual Formula", lecture_lines)

        # Assets
        formula = MathTex(r"x = \frac{\det(A_x)}{\det(A)}", font_size=48)
        area_a = Square(side_length=1.5, color=BLUE, fill_opacity=0.3)
        area_ax = Square(side_length=3.0, color=YELLOW, fill_opacity=0.3)

        # === Animation for Lecture Line 1 ===
        self.lecture_texts[0].set_color("#FFFFFF")
        self.place_at_grid(formula, 'B2', scale_factor=0.7)
        self.play(Write(formula))

        # === Animation for Lecture Line 2 ===
        self.lecture_texts[1].set_color("#FFFF00")
        self.place_at_grid(area_a, 'E1', scale_factor=0.8)
        self.place_at_grid(area_ax, 'E3', scale_factor=0.8)
        self.play(Create(area_a), Create(area_ax))

        # === Animation for Lecture Line 3 ===
        self.lecture_texts[2].set_color("#FF00FF")
        # Visualizing linear scaling: 1.5 * 2 = 3.0
        label_a = Text("Area A", font_size=18).next_to(area_a, DOWN)
        label_ax = Text("Area Ax", font_size=18).next_to(area_ax, DOWN)
        self.play(FadeIn(label_a), FadeIn(label_ax))

        # === Animation for Lecture Line 4 ===
        self.lecture_texts[3].set_color("#00FFFF")
        ratio = MathTex(r"\frac{3.0}{1.5} = 2", font_size=36, color=WHITE)
        self.place_at_grid(ratio, 'C2', scale_factor=0.75)
        self.play(Write(ratio))

        # === Animation for Lecture Line 5 ===
        self.lecture_texts[4].set_color("#FFFFFF")
        self.play(Indicate(formula))
        self.wait(2)
