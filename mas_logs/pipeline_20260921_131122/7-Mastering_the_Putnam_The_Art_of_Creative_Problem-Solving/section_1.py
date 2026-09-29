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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Putnam problems prioritize elegance over brute force.", "Standard math uses hammers for nuts.", "Putnam math analyzes structural stress points."]
        self.setup_layout("The Putnam Mindset: Beyond Standard Algorithms", lecture_lines)
        
        # Elements
        hammer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hammer.svg")
        nut = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/nut.svg")
        problem_symbol = RegularPolygon(n=5, color="#00CED1")
        solution_symbol = Star(n=5, color="#FF4500")
        summary_text = Text("Elegance wins.", font_size=32, color=WHITE)
        
        # === Animation for Lecture Line 1 ===
        # Putnam problems prioritize elegance over brute force.
        self.lecture[0].set_color("#FFD700")
        self.place_at_grid(problem_symbol, 'E3', scale_factor=0.6)
        self.place_at_grid(hammer, 'B2', scale_factor=0.3)
        self.play(FadeIn(problem_symbol), FadeIn(hammer))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Standard math uses hammers for nuts.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#00CED1")
        self.play(FadeOut(hammer))
        self.place_at_grid(nut, 'B2', scale_factor=0.3)
        self.play(FadeIn(nut))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Putnam math analyzes structural stress points.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FF4500")
        self.play(Transform(problem_symbol, solution_symbol))
        self.wait(1)
        
        self.play(FadeOut(problem_symbol), FadeOut(nut), FadeOut(self.lecture), FadeOut(self.title))
        self.place_in_area(summary_text, 'B4', 'F6', scale_factor=0.65)
        self.play(FadeIn(summary_text))
        self.wait(2)
