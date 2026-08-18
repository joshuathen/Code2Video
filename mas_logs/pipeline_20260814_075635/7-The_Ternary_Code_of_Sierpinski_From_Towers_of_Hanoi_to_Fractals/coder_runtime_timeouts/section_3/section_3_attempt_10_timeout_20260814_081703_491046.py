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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Visualizing the Sierpinski Triangle", [
            "State space forms the Sierpinski Triangle.",
            "Recursive moves build fractal geometry.",
            "Each turn adds a ternary digit.",
            "Path-finding ant traverses triangle edges.",
            "Fractal structure emerges from recursion."
        ])

        # Colors for highlights
        colors = ["#FF5733", "#33FF57", "#3357FF", "#F3FF33", "#FF33F3"]

        # Sierpinski Triangle construction
        def sierpinski_triangle(order, side_length):
            if order == 0:
                return Triangle(color=WHITE).scale(side_length)
            else:
                sub = sierpinski_triangle(order - 1, side_length / 2)
                t = VGroup(
                    sub.copy().shift(UP * side_length * 0.433),
                    sub.copy().shift(LEFT * side_length * 0.5 + DOWN * side_length * 0.433),
                    sub.copy().shift(RIGHT * side_length * 0.5 + DOWN * side_length * 0.433)
                )
                return t

        # === Animation for Lecture Line 1 ===
        triangle = sierpinski_triangle(3, 1.5)
        self.place_in_area(triangle, "B2", "E5")
        self.play(Create(triangle), self.lecture[0].animate.set_color(colors[0]))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(colors[1]))

        # === Animation for Lecture Line 3 ===
        digits = VGroup(*[Text(str(i), font_size=20, color=YELLOW) for i in [0, 1, 2]])
        self.place_at_grid(digits[0], "B2")
        self.place_at_grid(digits[1], "B5")
        self.place_at_grid(digits[2], "E3")
        self.play(FadeIn(digits), self.lecture[2].animate.set_color(colors[2]))

        # === Animation for Lecture Line 4 ===
        ant = Dot(color=RED).move_to(triangle.get_center())
        self.play(FadeIn(ant), self.lecture[3].animate.set_color(colors[3]))
        self.play(MoveAlongPath(ant, triangle), run_time=3)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(colors[4]))
        self.wait(1)