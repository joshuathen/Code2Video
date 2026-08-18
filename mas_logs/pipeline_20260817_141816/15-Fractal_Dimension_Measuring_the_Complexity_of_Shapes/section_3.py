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
        lecture_lines = [
            "Koch curves start with a simple segment.",
            "We replace parts to increase structural complexity.",
            "Calculating D gives us approximately 1.26.",
            "This fractional value indicates space-filling behavior.",
            "It is more complex than a standard line."
        ]
        self.setup_layout("Case Study: The Koch Curve", lecture_lines)
        
        def get_koch_segment(start, end):
            v = end - start
            p1 = start + v / 3
            p3 = start + 2 * v / 3
            rot = rotation_matrix(PI / 3, OUT)
            p2 = p1 + np.dot(rot, (v / 3))
            return [start, p1, p2, p3, end]

        def draw_koch(pts, depth):
            if depth == 0:
                return Line(pts[0], pts[-1], color="#F1C40F")
            lines = VGroup()
            for i in range(len(pts) - 1):
                new_pts = get_koch_segment(pts[i], pts[i+1])
                lines.add(draw_koch(new_pts, depth - 1))
            return lines

        start_pt = np.array([-1.5, 0, 0])
        end_pt = np.array([1.5, 0, 0])
        initial_line = Line(start_pt, end_pt, color="#F1C40F")
        self.place_in_area(initial_line, 'C2', 'F5', scale_factor=0.9)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#F1C40F")
        self.play(Create(initial_line))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#E67E22")
        koch_1 = draw_koch([start_pt, end_pt], 1)
        self.place_in_area(koch_1, 'B4', 'E6', scale_factor=0.7)
        self.play(ReplacementTransform(initial_line, koch_1))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#3498DB")
        koch_2 = draw_koch([start_pt, end_pt], 2)
        self.place_in_area(koch_2, 'C3', 'F6', scale_factor=0.8)
        self.play(ReplacementTransform(koch_1, koch_2))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#2ECC71")
        koch_3 = draw_koch([start_pt, end_pt], 3)
        self.place_in_area(koch_3, 'C3', 'F6', scale_factor=0.8)
        self.play(ReplacementTransform(koch_2, koch_3))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#9B59B6")
        d_text = Text("D ≈ 1.26", color="#9B59B6", font_size=36)
        snowflake_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/snowflake.svg")
        info = VGroup(d_text, snowflake_icon).arrange(RIGHT)
        self.place_at_grid(info, 'A4', scale_factor=0.8)
        self.play(Write(info), koch_3.animate.set_color("#9B59B6"))
        self.wait(2)
