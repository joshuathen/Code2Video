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
        lecture_lines = [
            "Dandelin spheres connect abstract and physical properties.",
            "Foci and directrix emerge from this logic.",
            "Ellipses model planetary orbits beautifully."
        ]
        self.setup_layout("Summary and Real-world Synthesis", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Display the Dandelin spheres and the conic section. (#FFFFFF)
        sphere1 = Sphere(radius=0.5, color=BLUE_D).set_opacity(0.6)
        sphere2 = Sphere(radius=0.8, color=BLUE_E).set_opacity(0.6)
        cone = Cone(base_radius=1.5, height=2.5, color=GRAY).rotate(PI/2, axis=RIGHT)
        conic = Ellipse(width=2, height=1, color=WHITE)
        
        dandelin_group = VGroup(sphere1, sphere2, cone, conic)
        self.place_in_area(dandelin_group, "A1", "C6", scale_factor=0.6)
        self.play(Create(dandelin_group))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        # Highlight the foci and directrices on the curve. (#FFFF00)
        focus1 = Dot(color=YELLOW).move_to(conic.get_center() + LEFT * 0.4)
        focus2 = Dot(color=YELLOW).move_to(conic.get_center() + RIGHT * 0.4)
        directrix = Line(UP, DOWN, color=YELLOW).scale(0.5).next_to(conic, RIGHT, buff=0.2)
        
        highlight_group = VGroup(focus1, focus2, directrix)
        self.play(Create(highlight_group))
        self.lecture[1].set_color("#FFFF00")

        # === Animation for Lecture Line 3 ===
        # Animate an ellipse representing a planetary orbit using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/planet.svg]. (#00FF00)
        planet = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/planet.svg")
        orbit = Ellipse(width=3, height=1.5, color=GREEN)
        
        self.place_at_grid(orbit, "E3", scale_factor=0.8)
        self.place_at_grid(planet, "E3", scale_factor=0.3)
        
        # Simple orbit animation
        self.play(Create(orbit), FadeIn(planet))
        self.play(MoveAlongPath(planet, orbit), run_time=3, rate_func=linear)
        self.lecture[2].set_color("#00FF00")
