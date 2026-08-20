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
        self.setup_layout("Summary and Synthesis", [
            "Beats create the pulse.",
            "Measures build the structure.",
            "Signatures act as blueprints.",
            "Geometric shapes show rhythmic variety.",
            "Rhythm follows clear mathematical rules."
        ])

        # Assets
        pulse = Dot(color=BLUE).scale(1.5)
        measure = Rectangle(height=1.5, width=2.5, color=GREEN)
        
        # Use asset from instructions
        blueprint = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/blueprint.svg")
        blueprint.set_color(WHITE)

        # Position assets based on critique and requirements
        self.place_at_grid(pulse, "B4", scale_factor=0.8)
        self.place_at_grid(measure, "D4", scale_factor=0.8)
        self.place_at_grid(blueprint, "E3", scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.play(Indicate(pulse))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        self.play(Indicate(measure))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.play(Indicate(blueprint))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(PURPLE))
        tri = Triangle(color=ORANGE).scale(0.5)
        sq = Square(color=RED).scale(0.5)
        shape_group = VGroup(tri, sq).arrange(RIGHT, buff=0.5)
        self.place_in_area(shape_group, "A5", "C6", scale_factor=0.9)
        self.play(FadeIn(shape_group))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.play(FadeOut(pulse), FadeOut(measure), FadeOut(blueprint), FadeOut(shape_group))
        self.wait(1)
