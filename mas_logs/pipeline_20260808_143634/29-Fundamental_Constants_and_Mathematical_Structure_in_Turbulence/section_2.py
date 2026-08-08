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

class Section2Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Kolmogorov hypothesized local isotropy in small scales.",
            "Energy cascades from large to small eddies.",
            "The cascade follows the famous 5/3 law.",
            "E(k) scales with dissipation to the two-thirds.",
            "Wavenumber k represents the vortex spatial frequency."
        ]
        self.setup_layout("The Kolmogorov Hypothesis (1941) & The 5/3 Law", lecture_lines)
        
        vortex_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/vortex.svg")
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.lecture[0]))
        kol_label = Text("Kolmogorov 1941", color="#FFD700")
        self.place_at_grid(kol_label, 'A2', scale_factor=0.7)
        self.place_at_grid(vortex_icon, 'A4', scale_factor=0.5)
        self.play(Write(kol_label), FadeIn(vortex_icon))
        self.lecture[0].set_color("#FFD700")

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(self.lecture[1]))
        axes = Axes(x_range=[0, 5, 1], y_range=[0, 5, 1], x_length=4, y_length=4)
        self.place_in_area(axes, 'B2', 'E4', scale_factor=0.6)
        self.play(Create(axes))
        self.lecture[1].set_color("#FF8C00")

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(self.lecture[2]))
        curve = axes.plot(lambda k: 1/((k+0.5)**(5/3)), color="#32CD33")
        self.play(Create(curve))
        self.lecture[2].set_color("#32CD33")

        # === Animation for Lecture Line 4 ===
        self.play(FadeIn(self.lecture[3]))
        inertial_box = DashedVMobject(Rectangle(width=1.5, height=2, color="#00BFFF"))
        self.place_in_area(inertial_box, 'F2', 'F4', scale_factor=0.5)
        self.play(Create(inertial_box))
        self.lecture[3].set_color("#00BFFF")

        # === Animation for Lecture Line 5 ===
        self.play(FadeIn(self.lecture[4]))
        self.lecture[4].set_color("#FF69B4")
        
        # Fade out everything except the 5/3 law curve and the vortex
        self.play(
            FadeOut(kol_label),
            FadeOut(axes),
            FadeOut(inertial_box),
            *[FadeOut(self.lecture[i]) for i in [0, 1, 3, 4]],
            run_time=2
        )
