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
        self.setup_layout("Conclusion and Summary", [
            "Distance is a flexible, human-defined measurement.",
            "Switching lenses reveals hidden number theory structures.",
            "-1 sits near all powers of 2."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Flash key terms: Valuation, Metric, Convergence.
        terms = VGroup(
            Text("Valuation", color=YELLOW),
            Text("Metric", color=BLUE),
            Text("Convergence", color=GREEN)
        ).arrange(DOWN, buff=0.5)
        self.place_at_grid(terms, 'C2', scale_factor=0.6)
        
        self.play(FadeIn(terms), self.lecture[0].animate.set_color(YELLOW))
        self.play(Flash(terms, color=WHITE, line_length=0.2, num_lines=12))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Display the final 2-adic number space.
        space = VGroup(*[Circle(radius=0.1, color=WHITE).shift(i*RIGHT*0.3 + j*UP*0.3) for i in range(3) for j in range(3)])
        self.place_at_grid(space, 'D3', scale_factor=0.8)
        
        self.play(Create(space), self.lecture[1].animate.set_color(BLUE))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Grid visual representation
        grid_visual = VGroup(*[Square(side_length=0.4, color=GRAY) for _ in range(4)])
        grid_visual.arrange(RIGHT, buff=0.1)
        self.place_in_area(grid_visual, 'A4', 'F6', scale_factor=0.7)

        self.play(FadeIn(grid_visual), self.lecture[2].animate.set_color(GREEN))
        self.wait(1)
        self.play(FadeOut(self.lecture), FadeOut(self.title), FadeOut(terms), FadeOut(space), FadeOut(grid_visual))
        self.wait(1)
