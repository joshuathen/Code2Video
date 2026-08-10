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
        self.setup_layout("Recording the Hologram (The Interference Pattern)", [
            "Split laser into two beams.",
            "Reference and Object beams interact.",
            "Interference pattern forms on film.",
            "This acts as complex grating.",
            "Grid stores 3D object data."
        ])
        
        # Assets
        film = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/film.svg", color=WHITE)
        obj_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/object.svg", color=YELLOW)
        
        ref_beam = Line(LEFT*1, RIGHT*1, color=BLUE)
        obj_beam = Line(DOWN*1, UP*1, color=GREEN)
        fringes = VGroup(*[Line(LEFT*0.5, RIGHT*0.5, color=WHITE).shift(UP*i*0.1) for i in range(-10, 11)])

        # === Animation for Lecture Line 1 ===
        # Split laser into two beams.
        self.lecture[0].set_color(BLUE)
        self.place_at_grid(ref_beam, 'B4', scale_factor=0.5)
        self.place_at_grid(obj_beam, 'E4', scale_factor=0.6)
        self.play(Create(ref_beam), Create(obj_beam))

        # === Animation for Lecture Line 2 ===
        # Reference and Object beams interact.
        self.lecture[1].set_color(GREEN)
        self.place_at_grid(obj_icon, 'D2', scale_factor=0.7)
        self.place_in_area(film, 'C3', 'E5', scale_factor=0.6)
        self.play(FadeIn(obj_icon), FadeIn(film))

        # === Animation for Lecture Line 3 ===
        # Interference pattern forms on film.
        self.lecture[2].set_color(WHITE)
        self.play(FadeIn(fringes.move_to(film.get_center()).scale(0.5)))

        # === Animation for Lecture Line 4 ===
        # This acts as complex grating.
        self.lecture[3].set_color(YELLOW)
        self.play(fringes.animate.set_stroke(opacity=0.5))

        # === Animation for Lecture Line 5 ===
        # Grid stores 3D object data.
        self.lecture[4].set_color(RED)
        self.wait(1)
