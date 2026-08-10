from manim import *
import numpy as np

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
        self.setup_layout("Conclusion: The Power of Abstraction", [
            "Abstraction unifies physics and engineering.",
            "We use familiar algebraic tools.",
            "Data points obey these same rules."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Using SVG asset
        vector = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg", color="#FF5733")
        self.place_at_grid(vector, 'B4', scale_factor=0.6)
        self.play(FadeIn(vector))
        self.lecture[0].set_color("#FF5733")

        # === Animation for Lecture Line 2 ===
        # Use updated grid positioning (B3, E5) and scale (0.35)
        dots = VGroup(*[Dot(point=np.random.uniform(-1, 1, 3), color="#3357FF") for _ in range(50)])
        self.place_in_area(dots, 'B3', 'E5', scale_factor=0.35)
        
        self.play(
            FadeOut(vector),
            Create(dots),
            run_time=2
        )
        self.lecture[1].set_color("#3357FF")

        # === Animation for Lecture Line 3 ===
        # Fade in a unified structure overlay
        overlay = VGroup(*[Line(dots[i].get_center(), dots[(i+1)%50].get_center(), color=WHITE, stroke_width=1) for i in range(50)])
        self.add(overlay)
        self.play(FadeIn(overlay))
        self.lecture[2].set_color(WHITE)
