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

class Section7Scene(TeachingScene):
    def construct(self):
        # Setup title and lecture lines
        lecture_lines = [
            "PDEs power weather forecasts and modern engineering designs.",
            "They are the mathematical foundation for describing physical reality.",
            "Mastering them unlocks the secrets of a changing world."
        ]
        self.setup_layout("Conclusion: The Language of the Universe", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Highlight first lecture line
        self.lecture[0].set_color(YELLOW)

        # Show icons: Smartphone [Asset], Rocket [Asset], and MRI
        # Load SVG assets
        smartphone_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/smartphone.svg").set_color(WHITE)
        rocket_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rocket.svg").set_color("#C0C0C0")
        
        # MRI Machine: No asset provided, create a representation with shapes
        mri_base = RoundedRectangle(height=1.2, width=1.0, corner_radius=0.1, color="#A9A9A9")
        mri_bore = Circle(radius=0.35, color="#A9A9A9").shift(UP*0.1)
        mri_icon = VGroup(mri_base, mri_bore)

        # Place icons in the grid
        self.place_at_grid(smartphone_icon, "B2", scale_factor=0.7)
        self.place_at_grid(rocket_icon, "B4", scale_factor=0.7)
        # Resolved Issue 41: Consistent scaling for mri_icon
        self.place_at_grid(mri_icon, "B6", scale_factor=0.7)

        self.play(FadeIn(smartphone_icon), FadeIn(rocket_icon), FadeIn(mri_icon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second lecture line
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        # Overlay yellow PDE formulas (#FFFF00) across the icons
        heat_eq = MathTex(r"\frac{\partial u}{\partial t} = \alpha \nabla^2 u", color="#FFFF00", font_size=24)
        wave_eq = MathTex(r"\frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u", color="#FFFF00", font_size=24)
        # Navier-Stokes equation
        navier_stokes = MathTex(r"\rho \frac{D\mathbf{v}}{Dt} = -\nabla p + \mu \nabla^2 \mathbf{v}", color="#FFFF00", font_size=18)

        self.place_at_grid(heat_eq, "C2")
        self.place_at_grid(wave_eq, "C4")
        # Resolved Issue 40: Use place_in_area for wide formula
        self.place_in_area(navier_stokes, "C5", "C6", scale_factor=1.0)

        self.play(Write(heat_eq), Write(wave_eq), Write(navier_stokes))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Highlight third lecture line
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        # Display 'PDEs: The Language of the Universe' (#FFFFFF) centered at the bottom
        final_text = Text("PDEs: The Language of the Universe", font_size=32, color=WHITE)
        self.place_in_area(final_text, "E1", "F6")

        self.play(FadeIn(final_text, shift=UP))
        self.wait(4)

        # Reset colors and final wait
        self.lecture[2].set_color(WHITE)
        self.wait(2)
