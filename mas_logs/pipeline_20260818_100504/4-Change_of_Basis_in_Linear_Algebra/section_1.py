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
        lecture_lines = [
            "Points in space are absolute.",
            "A coordinate system provides their numerical address.",
            "Changing the grid changes the coordinates.",
            "The point remains fixed in space.",
            "Perspective defines our numerical description."
        ]
        self.setup_layout("The Intuition: Points vs. Perspectives", lecture_lines)
        
        # Setup geometric elements
        origin = Dot(self.grid["F1"], color=WHITE)
        point_A = Dot(self.grid["B3"], color=WHITE)
        point_B = Dot(self.grid["D5"], color=WHITE)
        
        line_OA = Line(origin.get_center(), point_A.get_center(), color=WHITE)
        line_OB = Line(origin.get_center(), point_B.get_center(), color=WHITE)
        
        label_A = Text("A", font_size=20, color=WHITE).next_to(point_A, UP, buff=0.1)
        label_B = Text("B", font_size=20, color=WHITE).next_to(point_B, UP, buff=0.1)
        
        geometry_group = VGroup(origin, point_A, point_B, line_OA, line_OB, label_A, label_B)
        self.place_in_area(geometry_group, 'A2', 'F4', scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        # Points in space are absolute.
        self.lecture[0].set_color("#FFFFFF")
        self.play(FadeIn(point_A), FadeIn(point_B), Write(label_A), Write(label_B))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # A coordinate system provides their numerical address.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FF00FF")
        self.play(FadeIn(line_OA), FadeIn(line_OB), FadeIn(origin))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Changing the grid changes the coordinates.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#00FFFF")
        self.play(point_A.animate.set_color("#00FFFF"), point_B.animate.set_color("#00FFFF"))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # The point remains fixed in space.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color("#FFFF00")
        self.play(Flash(point_A), Flash(point_B))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Perspective defines our numerical description.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color("#00FF00")
        self.wait(1)
