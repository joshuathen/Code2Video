from manim import *

class Gear(VMobject):
    def __init__(self, outer_radius=1.0, inner_radius=0.7, number_of_teeth=12, color=WHITE, **kwargs):
        super().__init__(**kwargs)
        self.color = color
        self.number_of_teeth = number_of_teeth
        
        # Create gear shape using a polygon
        points = []
        for i in range(2 * number_of_teeth):
            angle = i * PI / number_of_teeth
            radius = outer_radius if i % 2 == 0 else inner_radius
            points.append([radius * np.cos(angle), radius * np.sin(angle), 0])
        
        self.set_points_smoothly(points)
        self.set_fill(color, opacity=1.0)
        self.set_stroke(WHITE, width=2)
        self.close_path()

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
        self.setup_layout("The Chain Rule: The Nested Function", [
            "Chain rule solves nested function derivatives.",
            "Think of opening a Matryoshka doll.",
            "Account for every layer of change."
        ])
        
        # === Animation for Lecture Line 1 ===
        gear_outer = Gear(outer_radius=1.2, inner_radius=0.8, number_of_teeth=12, color="#9B59B6")
        gear_inner = Gear(outer_radius=0.6, inner_radius=0.4, number_of_teeth=8, color="#9B59B6")
        doll = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/doll.svg")
        doll.set_color(WHITE)
        gears = VGroup(gear_outer, gear_inner, doll)
        
        self.place_in_area(gears, 'B3', 'C4', scale_factor=0.6)
        self.play(FadeIn(gears))
        self.lecture[0].set_color("#9B59B6")

        # === Animation for Lecture Line 2 ===
        gear_outer.set_color("#E91E63")
        self.play(Rotate(gear_outer, angle=2*PI, rate_func=linear))
        self.lecture[1].set_color("#E91E63")

        # === Animation for Lecture Line 3 ===
        formula = MathTex(r"f'(g(x)) \\cdot g'(x)", color="#1ABC9C")
        self.place_at_grid(formula, 'D3', scale_factor=0.85)
        self.play(Write(formula), run_time=2)
        self.lecture[2].set_color("#1ABC9C")
        self.wait(1)
