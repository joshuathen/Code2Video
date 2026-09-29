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
        self.setup_layout("The Origin of Refractive Index (n)", [
            "Refractive index n relates speed to light.",
            "Effective wave speed decreases in the medium.",
            "Dipole interference slows the wave significantly."
        ])
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        wave = SineWave(x_range=[-2, 2], amplitude=0.5, color="#00FFFF")
        self.place_at_grid(wave, 'B5', scale_factor=0.6)
        self.add(wave)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/electron.svg]
        dipole = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/electron.svg")
        dipole.set_color("#00FF00")
        self.place_at_grid(dipole, 'E4', scale_factor=0.8)
        self.play(FadeIn(dipole))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        retardation_label = Text("Phase Velocity Retardation", font_size=20, color="#FFFFFF")
        self.place_at_grid(retardation_label, 'F4', scale_factor=0.6)
        self.play(FadeIn(retardation_label))
        self.wait(2)

class SineWave(VMobject):
    def __init__(self, x_range, amplitude, **kwargs):
        super().__init__(**kwargs)
        self.set_points_smoothly([np.array([x, amplitude * np.sin(3*x), 0]) for x in np.linspace(x_range[0], x_range[1], 50)])
