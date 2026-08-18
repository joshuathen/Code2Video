from manim import *
import numpy as np

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
        self.setup_layout("Creating the 'Barber Pole' Effect", [
            "We use a polarized light and sugar solution.",
            "Light rotates differently based on its wavelength.",
            "This creates a helical, barber pole pattern.",
            "An analyzer reveals these vibrant color bands.",
            "The cylinder appears as a stack of colors."
        ])

        # Assets
        cylinder_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cylinder.svg")
        sugar_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sugar.svg")

        # === Animation for Lecture Line 1 ===
        # Show multiple stacked polarizers at slight angles in #00FFFF.
        polarizers = VGroup(*[Rectangle(width=2, height=0.5, color="#00FFFF") for _ in range(3)])
        polarizers.arrange(DOWN, buff=0.2)
        
        # Adding Asset Labeling (B018)
        self.place_at_grid(cylinder_icon, "B4", scale_factor=0.3)
        self.place_at_grid(polarizers, "B4", scale_factor=0.6)
        
        self.play(Create(polarizers), FadeIn(cylinder_icon))
        self.lecture[0].set_color("#00FFFF")

        # === Animation for Lecture Line 2 ===
        # Animate light passing through each, showing progressive rotation.
        light_beam = Line(start=ORIGIN, end=RIGHT*2, color="#FFFF00").rotate(0.1, about_point=ORIGIN)
        self.place_at_grid(light_beam, "C2", scale_factor=0.5)
        self.play(FadeIn(light_beam), light_beam.animate.rotate(0.5, about_point=light_beam.get_start()))
        self.lecture[1].set_color("#FFFF00")

        # === Animation for Lecture Line 3 ===
        # Highlight the resulting 'barber pole' spiral shape in #FF00FF.
        spiral = ParametricFunction(
            lambda t: np.array([np.sin(t*5), t, np.cos(t*5)]) * 0.5,
            t_range=np.array([0, 4]),
            color="#FF00FF"
        )
        
        # Adding Asset Labeling (B018)
        self.place_at_grid(sugar_icon, "E5", scale_factor=0.3)
        self.place_at_grid(spiral, "E5", scale_factor=0.7)
        
        self.play(Create(spiral), FadeIn(sugar_icon))
        self.lecture[2].set_color("#FF00FF")

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#00FF00")
        self.wait(1)
