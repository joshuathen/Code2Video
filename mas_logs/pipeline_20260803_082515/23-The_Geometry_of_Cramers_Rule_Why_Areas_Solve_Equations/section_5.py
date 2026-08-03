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
        # Initial Setup
        title_text = "Deriving the Rule: The Ratio of Areas"
        lecture_lines = [
            "The modified area equals x times original area.",
            "Thus, x is the ratio of these areas.",
            "This beautiful ratio is known as Cramer's Rule.",
            "We can find y using the same logic.",
            "Geometry transforms equations into simple ratios."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Visual: Display Area(b, v2) = x * Area(v1, v2)
        eq1 = MathTex(r"\text{Area}(\vec{b}, \vec{v}_2) = x \cdot \text{Area}(\vec{v}_1, \vec{v}_2)", font_size=32)
        self.place_in_area(eq1, "A1", "A6")
        
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(Write(eq1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Visual: x = Area(b, v2) / Area(v1, v2) in #FFFFFF
        eq2 = MathTex(r"x = \frac{\text{Area}(\vec{b}, \vec{v}_2)}{\text{Area}(\vec{v}_1, \vec{v}_2)}", font_size=32, color=WHITE)
        self.place_in_area(eq2, 'B1', 'B3', scale_factor=0.6)
        
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(YELLOW))
        self.play(TransformMatchingShapes(eq1.copy(), eq2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Visual: Show Area(b, v2) as the determinant of matrix A_1 in #FF00FF
        # x = det(A1) / det(A)
        eq3 = MathTex(r"x = \frac{\det(A_1)}{\det(A)}", font_size=36, color="#FF00FF")
        self.place_in_area(eq3, 'C1', 'C3', scale_factor=0.7)
        
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(YELLOW))
        self.play(Write(eq3))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Visual: Display the ratio y = Area(v1, b) / Area(v1, v2) in #FFFFFF
        # Show Area(v1, b) as the determinant of matrix A_2 in #00FFFF
        eq4_ratio = MathTex(r"y = \frac{\text{Area}(\vec{v}_1, \vec{b})}{\text{Area}(\vec{v}_1, \vec{v}_2)}", font_size=32, color=WHITE)
        eq4_det = MathTex(r"y = \frac{\det(A_2)}{\det(A)}", font_size=36, color="#00FFFF")
        self.place_in_area(eq4_ratio, 'B4', 'B6', scale_factor=0.6)
        self.place_in_area(eq4_det, 'C4', 'C6', scale_factor=0.7)

        self.play(self.lecture[2].animate.set_color(WHITE), self.lecture[3].animate.set_color(YELLOW))
        self.play(Write(eq4_ratio))
        self.wait(0.5)
        self.play(Write(eq4_det))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Visual: Highlight the common denominator det(A) in both formulas in #FFFF00
        highlight_x = SurroundingRectangle(eq3, color="#FFFF00", buff=0.1)
        highlight_y = SurroundingRectangle(eq4_det, color="#FFFF00", buff=0.1)
        common_label = Text("det(A) is the shared basis", font_size=24, color="#FFFF00")
        # Position common_label closer (row D)
        self.place_in_area(common_label, "D1", "D6")

        self.play(self.lecture[3].animate.set_color(WHITE), self.lecture[4].animate.set_color(YELLOW))
        self.play(
            Create(highlight_x),
            Create(highlight_y),
            Write(common_label)
        )
        self.wait(3)
