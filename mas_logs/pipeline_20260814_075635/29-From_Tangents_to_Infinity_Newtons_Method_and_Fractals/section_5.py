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
        self.setup_layout("Conclusion and Intuition", [
            "Simple rules create infinite, beautiful complexity.",
            "Chaos thrives at the meeting of basins.",
            "Newton's method reveals structure within the unknown."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Simple rules create infinite, beautiful complexity.
        circle = Circle(radius=1.0, color=BLUE).set_stroke(width=4)
        dot = Dot(color=YELLOW)
        dot.move_to(circle.get_right())
        circle_group = VGroup(circle, dot)
        
        # Fixing layout issues as requested
        self.place_in_area(circle_group, 'A4', 'B6', scale_factor=0.5)
        
        self.play(Create(circle), Write(dot))
        self.play(self.lecture[0].animate.set_color("#00FFFF"), run_time=1)
        
        # === Animation for Lecture Line 2 ===
        # Chaos thrives at the meeting of basins.
        line1 = Line(start=self.grid['B2'], end=self.grid['E5'], color=RED)
        line2 = Line(start=self.grid['B5'], end=self.grid['E2'], color=GREEN)
        
        self.play(Create(line1), Create(line2))
        self.play(self.lecture[1].animate.set_color("#FF00FF"), run_time=1)
        
        # === Animation for Lecture Line 3 ===
        # Newton's method reveals structure within the unknown.
        final_text = Text("Newton's Wisdom", font_size=36, color=YELLOW)
        self.place_at_grid(final_text, 'E4', scale_factor=0.9)
        
        self.play(Write(final_text))
        self.play(self.lecture[2].animate.set_color("#FFFF00"), run_time=1)
        self.wait(2)
