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
        self.setup_layout("Introduction: The Intuitive Hook", [
            "Can we always find a square on any closed loop?",
            "Think of a tangled rubber band on a table.",
            "Pick four points to form a perfect square."
        ])

        # Assets
        table = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/table.svg")
        self.place_in_area(table, "A1", "F6", scale_factor=1.5)
        self.add(table)

        # Closed curve
        curve = VMobject(color=WHITE)
        curve.set_points_smoothly([
            [1.5, 0.5, 0], [2.5, 1.5, 0], [3.5, 0.5, 0], [2.5, -0.5, 0], [1.5, 0.5, 0]
        ])
        curve_label = Text("Curve", font_size=20, color=WHITE)
        curve_label.next_to(curve, UP, buff=0.1)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        self.play(Create(curve), Write(curve_label))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color("#FF00FF"))
        
        # Four points moving on the curve (represented as dots)
        dots = VGroup(*[Dot(color="#FF00FF") for _ in range(4)])
        for i, dot in enumerate(dots):
            dot.move_to(curve.point_from_proportion(i/4))
        self.add(dots)
        self.play(Rotating(dots, about_point=curve.get_center(), run_time=2))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color("#00FFFF"))
        
        # Form a square
        square_points = [
            np.array([2.0, 0.5, 0]), np.array([3.0, 0.5, 0]),
            np.array([3.0, -0.5, 0]), np.array([2.0, -0.5, 0])
        ]
        square = Polygon(*square_points, color="#00FFFF")
        self.play(Create(square))
        self.wait(1)
