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
        self.setup_layout("Summary & Visual Connection", [
            "Span, independence, and basis are core building blocks.",
            "Losing a basis vector collapses the space.",
            "A 3D crane loses movement if vectors dependent."
        ])
        
        # Colors for highlighting
        c1, c2, c3 = "#FFD700", "#00CED1", "#FF4500"
        
        # === Animation for Lecture Line 1 ===
        # Representing Span, Independence, Basis
        span_obj = Circle(radius=0.5, color=c1, fill_opacity=0.3)
        indep_obj = VGroup(*[Arrow(ORIGIN, UP*0.5, color=c2), Arrow(ORIGIN, RIGHT*0.5, color=c2)])
        basis_obj = VGroup(span_obj.copy().scale(0.5), indep_obj.copy().scale(0.5))
        
        self.place_at_grid(span_obj, 'B2')
        self.place_at_grid(indep_obj, 'B4')
        self.place_at_grid(basis_obj, 'A5', scale_factor=0.7)
        
        self.play(FadeIn(span_obj), FadeIn(indep_obj), FadeIn(basis_obj))
        self.lecture[0].set_color(c1)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Collapse visualization
        plane = Square(side_length=2, color=WHITE, fill_opacity=0.1)
        self.place_in_area(plane, 'D4', 'F6', scale_factor=0.5)
        self.play(FadeIn(plane))
        
        # Shrink the plane
        self.play(plane.animate.stretch(0.1, dim=1), run_time=2)
        self.lecture[1].set_color(c2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Use SVG asset
        crane = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/crane.svg")
        self.place_at_grid(crane, 'E4', scale_factor=0.6)
        self.play(FadeIn(crane))
        self.play(crane.animate.rotate(PI/4, about_point=crane.get_center()))
        self.lecture[2].set_color(c3)
        self.wait(2)
