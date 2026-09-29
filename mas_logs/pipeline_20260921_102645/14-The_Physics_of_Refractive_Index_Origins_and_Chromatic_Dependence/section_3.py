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
            "Atoms have specific natural frequencies.",
            "Higher frequencies interact more strongly.",
            "Refractive index changes with wavelength.",
            "This phenomenon is called dispersion.",
            "Blue light oscillates tighter than red."
        ]
        self.setup_layout("The Dependence on Color: Dispersion", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        atoms = VGroup(*[Circle(radius=0.2, color=BLUE).move_to(self.grid[pos]) for pos in ['B2', 'B4', 'C3', 'D2', 'D4']])
        self.play(Create(atoms))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF4500"))
        # Using SVGMobject for asset
        prism = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/prism.svg", color=WHITE)
        self.place_at_grid(prism, 'B5', scale_factor=0.6)
        self.play(FadeIn(prism))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00CED1"))
        axes = Axes(x_range=[0, 5], y_range=[1, 3], axis_config={"include_tip": False}).scale(0.4)
        self.place_in_area(axes, 'D2', 'E4', scale_factor=0.7)
        self.play(Create(axes))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#ADFF2F"))
        # Use wave animation as requested in critique
        wave_blue = Wave(freq=4).set_color(BLUE).scale(0.3)
        wave_red = Wave(freq=1).set_color(RED).scale(0.3)
        wave_animation = VGroup(wave_blue, wave_red).arrange(DOWN)
        self.place_at_grid(wave_animation, 'B3', scale_factor=0.7)
        self.play(Create(wave_animation))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#1E90FF"))
        self.wait(1)

class Wave(VMobject):
    def __init__(self, freq=1, **kwargs):
        super().__init__(**kwargs)
        self.set_points_smoothly([
            np.array([x, np.sin(freq * x * PI), 0]) for x in np.linspace(-1, 1, 50)
        ])
