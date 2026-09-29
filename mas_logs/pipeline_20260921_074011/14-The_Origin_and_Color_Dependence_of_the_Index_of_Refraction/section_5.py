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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Conclusion and Real-world Application", [
            "Color dependence is inherent to matter.", 
            "Chromatic aberration affects lens performance.", 
            "Achromatic doublets cancel chromatic dispersion."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Show refractive index formula summary #FFFFFF.
        formula = MathTex(r"n(\omega) \approx 1 + \sum \frac{f_j \omega_p^2}{\omega_j^2 - \omega^2}", color=WHITE)
        self.place_in_area(formula, 'B3', 'B6', scale_factor=0.9)
        self.play(Write(formula))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Display lens [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/lens.svg] refracting light rays #00FF00.
        lens = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/lens.svg")
        self.place_at_grid(lens, 'D2', scale_factor=0.7)
        
        ray1 = Line(start=np.array([-2, 1, 0]), end=np.array([0, 1, 0]), color=RED)
        ray2 = Line(start=np.array([-2, 0.5, 0]), end=np.array([0, 0.5, 0]), color=BLUE)
        
        self.play(FadeIn(lens), Create(ray1), Create(ray2))
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight real-world optical glass [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/glass.svg] application #FFD700.
        doublet = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/glass.svg")
        self.place_at_grid(doublet, 'D5', scale_factor=0.7)
        
        self.play(FadeIn(doublet))
        self.play(self.lecture[2].animate.set_color("#FFD700"))
        self.wait(2)
