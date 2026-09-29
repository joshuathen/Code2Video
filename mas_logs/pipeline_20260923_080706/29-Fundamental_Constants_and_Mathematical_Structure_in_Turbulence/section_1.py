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
            "Turbulence is a multi-scale, chaotic flow.",
            "Energy injects at large, stable scales.",
            "Eddies break down, transferring energy downward.",
            "Smallest scales dissipate energy as heat.",
            "This is Richardson’s energy cascade."
        ]
        self.setup_layout("Introduction: The Chaos in the Coffee Cup", lecture_lines)
        
        # Use SVG asset as requested in Issue 16
        cup = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coffee.svg", color=WHITE)
        self.place_in_area(cup, 'C2', 'F5', scale_factor=0.7)
        
        vectors = VGroup()

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(Create(cup))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(BLUE)
        v1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coffee.svg", color=BLUE)
        self.place_at_grid(v1, 'C6', scale_factor=0.6)
        self.play(FadeIn(v1))
        vectors.add(v1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(BLUE)
        v2 = Arrow(start=LEFT, end=RIGHT, color=BLUE).scale(0.3)
        self.place_at_grid(v2, 'D6', scale_factor=0.6)
        self.play(FadeIn(v2))
        vectors.add(v2)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(RED)
        v3 = Dot(color=RED).scale(2)
        self.place_at_grid(v3, 'E6', scale_factor=0.6)
        self.play(FadeIn(v3))
        vectors.add(v3)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(ORANGE)
        self.play(Flash(vectors.get_center(), color=ORANGE))
        self.wait(1)
