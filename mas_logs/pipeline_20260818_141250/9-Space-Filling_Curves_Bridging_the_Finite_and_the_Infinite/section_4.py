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
        self.setup_layout("Conclusion: The Beauty of Convergent Complexity", [
            "Finite rules generate infinite complexity.",
            "Topology bridges lines and squares.",
            "Mathematical beauty emerges from recursion."
        ])
        
        # Assets
        recursive_tree = VGroup(
            *[Line(ORIGIN, UP * 0.5).rotate(i * 360/8 * DEGREES) for i in range(8)]
        ).set_color(BLUE)
        
        final_square = Square(side_length=2, color=ORANGE).set_fill(ORANGE, opacity=0.3)
        paradox_dim = Text("1D vs 2D", font_size=36, color=YELLOW)

        # === Animation for Lecture Line 1 ===
        self.place_in_area(recursive_tree, 'B3', 'C4', scale_factor=0.8)
        self.play(FadeIn(recursive_tree))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(FadeOut(recursive_tree))
        self.place_at_grid(final_square, 'D4', scale_factor=0.6)
        self.play(Create(final_square))
        self.play(self.lecture[1].animate.set_color("#FF4500"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.place_at_grid(paradox_dim, 'E4', scale_factor=0.9)
        self.play(FadeIn(paradox_dim))
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        self.wait(1)
