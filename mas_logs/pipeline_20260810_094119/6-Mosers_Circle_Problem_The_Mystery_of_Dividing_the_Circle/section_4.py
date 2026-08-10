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
        self.setup_layout("The Logic: Combinatorial Geometry", [
            "Regions depend on chord intersections.",
            "Four points define one intersection.",
            "Formula: C(n,4) plus C(n,2) plus one. [Asset: logic_animation]"
        ])
        
        # === Animation for Lecture Line 1 ===
        # Regions depend on chord intersections.
        circle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg")
        self.place_in_area(circle, 'B2', 'D4', scale_factor=0.6)
        self.play(Create(circle))
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        # Four points define one intersection.
        dots = VGroup(*[Dot(radius=0.08, color="#FFFF00") for _ in range(4)])
        # Positioning dots relative to SVG bounding box
        dots[0].move_to(circle.get_right() - 0.2*UP)
        dots[1].move_to(circle.get_top() - 0.2*RIGHT)
        dots[2].move_to(circle.get_left() + 0.2*DOWN)
        dots[3].move_to(circle.get_bottom() + 0.2*LEFT)
        
        chord1 = Line(dots[0].get_center(), dots[2].get_center(), color="#00FFFF")
        chord2 = Line(dots[1].get_center(), dots[3].get_center(), color="#00FFFF")
        intersection_point = line_intersection(
            (chord1.get_start(), chord1.get_end()),
            (chord2.get_start(), chord2.get_end())
        )
        intersection_dot = Dot(color="#FFFF00").move_to(intersection_point)
        
        self.play(Create(dots), Create(chord1), Create(chord2), Create(intersection_dot))
        self.lecture[1].set_color(RED)

        # === Animation for Lecture Line 3 ===
        # Formula: C(n,4) plus C(n,2) plus one.
        formula = MathTex(r"C(n,4) + C(n,2) + 1", color=GOLD)
        self.place_at_grid(formula, 'E3', scale_factor=1.0)
        
        self.play(Write(formula))
        self.lecture[2].set_color(GOLD)
        
        # Highlighting total regions
        regions_count = Text("11", font_size=40, color="#00FF00")
        self.place_at_grid(regions_count, 'E4', scale_factor=0.9)
        self.play(Indicate(regions_count))
        self.wait(2)
