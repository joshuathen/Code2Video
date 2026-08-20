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
        self.setup_layout("The Structural Formula", [
            "Euler's formula connects vertices and edges.",
            "Region count uses intersections and chords.",
            "The formula is C(n,4) plus C(n,2) plus one."
        ])
        
        # Define mobjects
        formula = MathTex(r"R = \binom{n}{4} + \binom{n}{2} + 1", font_size=42, color="#3498DB")
        label_4 = MathTex(r"\binom{n}{4}", font_size=32, color="#E74C3C")
        label_2 = MathTex(r"\binom{n}{2}", font_size=32, color="#E74C3C")
        label_1 = MathTex(r"1", font_size=32, color="#E74C3C")
        components = VGroup(label_4, label_2, label_1).arrange(RIGHT, buff=0.5)
        
        # Placeholder for grid visual requested by critic
        grid_visual = Square(side_length=2, color=GRAY)
        
        # Apply critic-suggested positioning
        self.place_in_area(formula, 'B4', 'C6', scale_factor=0.9)
        self.place_at_grid(components, 'D4', scale_factor=0.7)
        self.place_in_area(grid_visual, 'A4', 'F6', scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#3498DB"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#E74C3C"))
        self.play(FadeIn(formula))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#2ECC71"))
        self.play(FadeIn(components))
        self.wait(2)
