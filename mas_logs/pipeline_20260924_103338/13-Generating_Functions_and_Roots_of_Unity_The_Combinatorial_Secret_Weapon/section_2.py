from manim import *
import numpy as np

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

class Section2Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "We need to sum specific coefficients.",
            "Roots of unity form regular polygons.",
            "[Asset: RootsUnitCircle] shows these complex pointers.",
            "Multiplying by ω acts as a filter.",
            "This sieve cancels out unwanted terms."
        ]
        self.setup_layout("The Geometry of Selection: Complex Roots of Unity", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Algebraic sum placeholder
        sum_eq = MathTex(r"\sum_{j=0}^{n-1} a_j", color="#00CCFF")
        self.place_at_grid(sum_eq, "C5", scale_factor=0.8)
        self.play(Write(sum_eq))
        self.lecture[0].set_color("#00CCFF")

        # === Animation for Lecture Line 2 ===
        # Roots of unity (regular polygon)
        # B002: Avoid A/F rows. B002: scale 0.6-0.7.
        circle = Circle(radius=1.0, color="#FFFFFF")
        roots = VGroup(*[Dot(point=np.array([np.cos(2*PI*k/3), np.sin(2*PI*k/3), 0])) for k in range(3)])
        roots.set_color("#FFCC00")
        poly = Polygon(*[r.get_center() for r in roots], color="#FFCC00")
        visual_group = VGroup(circle, poly, roots)
        self.place_in_area(visual_group, "B4", "E6", scale_factor=0.6)
        
        self.play(Create(circle), Create(poly), FadeIn(roots))
        self.lecture[1].set_color("#FFCC00")

        # === Animation for Lecture Line 3 ===
        # Asset: RootsUnitCircle (repurposing visual_group)
        vector = Line(start=ORIGIN, end=roots[0].get_center(), color="#FFFFFF")
        omega_label = MathTex(r"\omega", color="#FFFFFF")
        self.place_at_grid(omega_label, "D4", scale_factor=0.7)
        self.play(Create(vector), Write(omega_label))
        self.lecture[2].set_color("#FFFFFF")

        # === Animation for Lecture Line 4 ===
        # Filter icon: [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/filter.svg]
        filter_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/filter.svg", color="#FF3399")
        self.place_at_grid(filter_icon, "C6", scale_factor=0.5)
        self.play(FadeIn(filter_icon))
        self.lecture[3].set_color("#FF3399")

        # === Animation for Lecture Line 5 ===
        # Sieve cancellation (shift colors)
        self.play(
            roots.animate.set_color("#FF0000"),
            sum_eq.animate.set_color("#FF0000"),
            run_time=2
        )
        self.lecture[4].set_color("#FF0000")
