from manim import *

class SierpinskiTriangle(VGroup):
    def __init__(self, order=3, **kwargs):
        super().__init__(**kwargs)
        # Simplified Sierpinski construction using lines to be lighter
        points = [LEFT+DOWN, RIGHT+DOWN, UP]
        self.add(Polygon(*points, color=BLUE))
        self.order = order
        for _ in range(order):
            new_triangles = VGroup()
            for tri in self:
                vertices = tri.get_vertices()
                p1, p2, p3 = vertices[0], vertices[1], vertices[2]
                m1 = (p1 + p2) / 2
                m2 = (p2 + p3) / 2
                m3 = (p3 + p1) / 2
                new_triangles.add(Polygon(p1, m1, m3, color=BLUE))
                new_triangles.add(Polygon(m1, p2, m2, color=BLUE))
                new_triangles.add(Polygon(m3, m2, p3, color=BLUE))
            self.submobjects = new_triangles.submobjects

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
        lecture_lines = [
            "Ternary numbers serve as precise path coordinates.",
            "Incrementing ternary values dictates optimal movement sequences.",
            "This recursive harmony guides paths through the fractal."
        ]
        self.setup_layout("The Synthesis: Recursive Harmony", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Overlay triangle structure onto [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/tower.svg]
        tower = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/tower.svg")
        triangle = SierpinskiTriangle(order=3)
        
        self.place_in_area(tower, 'E1', 'F2', scale_factor=0.4)
        self.place_in_area(triangle, 'D2', 'F5', scale_factor=0.6)
        
        self.play(FadeIn(tower))
        self.play(Create(triangle))
        self.lecture[0].set_color("#00FFFF")

        # === Animation for Lecture Line 2 ===
        # Animate recursive moves following the Sierpinski path
        dot = Dot(color=YELLOW)
        self.place_at_grid(dot, 'B5', scale_factor=0.7)
        self.play(FadeIn(dot))
        
        path = VMobject().set_points_smoothly([self.grid['B5'], self.grid['D3'], self.grid['C4'], self.grid['B5']])
        path.set_color(YELLOW)
        
        self.play(MoveAlongPath(dot, path), run_time=2)
        self.lecture[1].set_color("#00FFFF")

        # === Animation for Lecture Line 3 ===
        # Synthesize final state
        self.play(triangle.animate.set_color("#00FFFF"), dot.animate.set_color("#00FFFF"), tower.animate.set_opacity(0.3))
        self.lecture[2].set_color("#00FFFF")
        self.wait(2)
