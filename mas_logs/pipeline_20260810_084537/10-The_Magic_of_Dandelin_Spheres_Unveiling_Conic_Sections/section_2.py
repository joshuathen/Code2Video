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

class Section2Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Dandelin spheres fit perfectly inside a cone.",
            "They are tangent to the cone in a circle.",
            "Each sphere touches the cutting plane at one point.",
            "These tangency points are the section's focal points."
        ]
        self.setup_layout("Introducing Dandelin Spheres", lecture_lines)
        
        # Mobjects using assets
        # Asset paths:
        # /scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg
        # /scratch/pawsey1357/jthen/Code2Video/assets/icon/cone.svg
        s1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg").set_color(YELLOW)
        s2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg").set_color(YELLOW)
        cone = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cone.svg").set_color(WHITE)
        
        l1 = Text("S1", font_size=18, color=YELLOW)
        l2 = Text("S2", font_size=18, color=YELLOW)
        
        s1_group = VGroup(s1, l1)
        s2_group = VGroup(s2, l2)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.place_at_grid(cone, "C4", scale_factor=0.8)
        self.place_at_grid(s1_group, "B3", scale_factor=0.8)
        self.place_at_grid(s2_group, "E4", scale_factor=0.8)
        self.play(FadeIn(cone), FadeIn(s1_group), FadeIn(s2_group))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        c1 = Circle(radius=0.5, color=BLUE).set_stroke(width=2)
        c2 = Circle(radius=0.3, color=BLUE).set_stroke(width=2)
        self.place_at_grid(c1, "B2", scale_factor=0.6)
        self.place_at_grid(c2, "E5", scale_factor=0.6)
        self.play(Create(c1), Create(c2))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        p1 = Dot(color=RED)
        p2 = Dot(color=RED)
        self.place_at_grid(p1, "C3", scale_factor=0.5)
        self.place_at_grid(p2, "D5", scale_factor=0.5)
        self.play(FadeIn(p1), FadeIn(p2))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(YELLOW))
        self.play(Indicate(l1), Indicate(l2))
