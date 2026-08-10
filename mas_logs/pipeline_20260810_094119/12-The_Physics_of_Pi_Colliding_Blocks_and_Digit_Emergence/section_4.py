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
        self.setup_layout("Scaling Powers of 100", [
            "Increase mass ratio by powers of 100.",
            "Observe collision counts matching Pi digits.",
            "Mass ratio 100 yields 3 collisions.",
            "10,000 ratio yields 31 collisions.",
            "1,000,000 ratio yields 314 collisions."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Increase mass ratio by powers of 100.
        expr = MathTex("M/m = 100^n", font_size=48, color="#ADD8E6")
        self.place_in_area(expr, 'A3', 'C5', scale_factor=0.8)
        self.play(Write(expr))
        self.lecture[0].set_color("#ADD8E6")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Observe collision counts matching Pi digits.
        self.lecture[1].set_color("#FFFFE0")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Mass ratio 100 yields 3 collisions.
        self.lecture[2].set_color("#FFB6C1")
        self.play(expr.animate.scale(0.8)) # Minor visual emphasis
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # 10,000 ratio yields 31 collisions.
        new_expr = MathTex("10000", font_size=60, color="#FFFF00")
        self.place_in_area(new_expr, 'D3', 'F5', scale_factor=0.8)
        self.lecture[3].set_color("#FFFF00")
        self.play(Write(new_expr))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # 1,000,000 ratio yields 314 collisions.
        final_expr = MathTex("1000000", font_size=72, color="#98FB98")
        self.place_in_area(final_expr, 'A4', 'F6', scale_factor=1.0)
        self.lecture[4].set_color("#98FB98")
        self.play(ReplacementTransform(new_expr, final_expr))
        self.wait(2)
