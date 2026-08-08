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
            "Pi emerges from the geometry.", 
            "Physics and math are deeply linked.", 
            "Counting collisions calculates Pi."
        ])
        
        # Elements
        pi_symbol = MathTex(r"\pi", font_size=96, color=WHITE)
        geometry_label = Text("Geometry", font_size=32, color=WHITE)
        
        physics_label = Text("Physics", font_size=32, color=WHITE)
        math_label = Text("Math", font_size=32, color=WHITE)
        connection_line = Line(start=ORIGIN, end=RIGHT*2, color=WHITE)
        
        collision_text = Text("Collisions", font_size=40, color=WHITE)
        equals_sign = MathTex(r"=", font_size=60, color=WHITE)
        pi_res = MathTex(r"\pi", font_size=60, color=WHITE)
        result_group = VGroup(collision_text, equals_sign, pi_res).arrange(RIGHT)

        # Place elements
        self.place_in_area(VGroup(pi_symbol, geometry_label).arrange(DOWN), 'A1', 'C3', scale_factor=0.8)
        self.place_in_area(VGroup(physics_label, connection_line, math_label).arrange(RIGHT), 'D1', 'E6', scale_factor=0.8)
        self.place_in_area(result_group, 'F1', 'F6', scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"), 
                  pi_symbol.animate.set_color("#FFFFFF"),
                  geometry_label.animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"), 
                  physics_label.animate.set_color("#00FFFF"),
                  math_label.animate.set_color("#00FFFF"),
                  connection_line.animate.set_color("#00FFFF"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"), 
                  collision_text.animate.set_color("#00FF00"),
                  pi_res.animate.set_color("#00FF00"))
        self.wait(2)
