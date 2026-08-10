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
        self.setup_layout("Synthesis & Summary", [
            "- Integration reverses differentiation.",
            "- They are inverse mathematical processes.",
            "- Switch perspectives to solve problems."
        ])
        
        # Elements
        table = VGroup(
            Text("Derivative: Slope / Rate", font_size=24, color="#00BFFF"),
            Text("Integral: Area / Total", font_size=24, color="#32CD32"),
            Text("Inverse: Fundamental Theorem", font_size=24, color=WHITE)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        
        arrow = Arrow(start=UP, end=DOWN, color=WHITE)
        summary_group = VGroup(table, arrow)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00BFFF"))
        self.place_in_area(table, 'B2', 'C5', scale_factor=0.6)
        self.play(Write(table))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#32CD32"))
        self.place_at_grid(arrow, 'D4', scale_factor=0.7)
        self.play(GrowArrow(arrow))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.place_in_area(summary_group, 'E3', 'F6', scale_factor=0.65)
        self.play(FadeOut(table), FadeOut(arrow))
        self.wait(1)
