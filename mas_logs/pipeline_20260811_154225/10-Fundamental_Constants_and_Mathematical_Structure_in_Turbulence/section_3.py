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
        lecture_lines = [
            "Energy spectral density follows a power law.",
            "The equation relates energy to wavenumber.",
            "E equals C times epsilon to power.",
            "Wavenumber k shows a negative five-thirds slope.",
            "C represents the universal Kolmogorov constant."
        ]
        self.setup_layout("Mathematical Structure: The Kolmogorov Constant (C)", lecture_lines)
        
        # --- Animation Objects ---
        const_c = Text("C", font_size=72, color="#00FF00")
        # Fixed: SVG files should be loaded using SVGMobject, not ImageMobject
        icon_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        axes = NumberPlane(
            x_range=[0, 6, 1], y_range=[0, 6, 1],
            x_length=4, y_length=4,
            axis_config={"include_tip": True}
        )
        # Power law function E(k) ~ k^(-5/3) roughly slope -1.67
        curve = axes.plot(lambda x: 4 * (x + 0.5)**(-1.67), color="#FF00FF")
        equation = MathTex(r"E(k) = C \cdot \epsilon^{2/3} \cdot k^{-5/3}", font_size=32)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FF00")
        self.place_at_grid(const_c, "B4", scale_factor=0.8)
        self.place_at_grid(icon_asset, "B5", scale_factor=0.5)
        self.play(FadeIn(const_c), FadeIn(icon_asset))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF00FF")
        self.place_at_grid(equation, "C4", scale_factor=0.9)
        self.play(FadeIn(equation))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FF00")
        self.play(Indicate(const_c), Indicate(equation[0][7:8])) # Highlight C
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FF00FF")
        self.place_in_area(axes, "D3", "F5", scale_factor=0.6)
        self.play(Create(axes), Create(curve))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#00FF00")
        self.play(Circumscribe(const_c), Circumscribe(curve))
        self.wait(2)
