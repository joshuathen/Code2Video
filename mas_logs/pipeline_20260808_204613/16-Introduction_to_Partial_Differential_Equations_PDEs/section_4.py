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
        self.setup_layout("Boundary and Initial Conditions", [
            "Equations need rules to function.",
            "Initial conditions set the start.",
            "Boundary conditions define the edges."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Show a grid representing the spatial domain
        domain = Rectangle(width=4, height=4, color=WHITE)
        self.place_in_area(domain, "B4", "E6", scale_factor=0.5)
        self.play(Create(domain))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        # Show the initial state wave as an input to the PDE using asset
        # Use ImageMobject or SVGMobject for asset
        wave = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wave.svg", color=GREEN)
        self.place_in_area(wave, "B3", "E6", scale_factor=0.7)
        self.play(FadeIn(wave))
        self.lecture[1].set_color(GREEN)

        # === Animation for Lecture Line 3 ===
        # Highlight boundary edges in #3357FF
        edges = VGroup(
            Line(domain.get_corner(UL), domain.get_corner(UR), color="#3357FF", stroke_width=6),
            Line(domain.get_corner(UR), domain.get_corner(DR), color="#3357FF", stroke_width=6),
            Line(domain.get_corner(DR), domain.get_corner(DL), color="#3357FF", stroke_width=6),
            Line(domain.get_corner(DL), domain.get_corner(UL), color="#3357FF", stroke_width=6)
        )
        self.play(Create(edges))
        self.lecture[2].set_color("#3357FF")
        self.wait(2)
