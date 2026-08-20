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
            "Reference beams provide the stable baseline.",
            "Object beams carry scattered phase information.",
            "Recording combines them into interference patterns.",
            "Holograms freeze this grid on film.",
            "The pattern stores complete wave geometry."
        ]
        self.setup_layout("The Holographic Recording Process", lecture_lines)
        
        # Load assets
        laser = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg", color="#00BFFF")
        film = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/film.svg", color="#FF69B4")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_opacity(1)
        ref_beam = laser.copy()
        obj_beam = laser.copy()
        self.place_at_grid(ref_beam, "A4", scale_factor=0.6)
        self.place_at_grid(obj_beam, "B4", scale_factor=0.6)
        self.play(FadeIn(ref_beam), FadeIn(obj_beam))
        self.play(self.lecture[0].animate.set_color("#00BFFF"))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_opacity(1)
        self.play(self.lecture[1].animate.set_color("#00BFFF"))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_opacity(1)
        self.place_at_grid(film, "D5", scale_factor=0.7)
        self.play(FadeIn(film))
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_opacity(1)
        intersection = Cross(color="#FFD700").scale(0.3)
        self.place_at_grid(intersection, "C3")
        self.play(FadeIn(intersection))
        self.play(self.lecture[3].animate.set_color("#FFD700"))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_opacity(1)
        fringes = VGroup(*[Line(UP*0.5, DOWN*0.5, color="#32CD32").shift(RIGHT*i*0.05) for i in range(10)])
        self.place_at_grid(fringes, "F4", scale_factor=0.5)
        self.play(FadeIn(fringes))
        self.play(self.lecture[4].animate.set_color("#32CD32"))
        self.wait(2)
