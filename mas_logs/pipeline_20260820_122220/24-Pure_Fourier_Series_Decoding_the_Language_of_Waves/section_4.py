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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Calculating Coefficients: The Detective Work", [
            "We extract coefficients using integral filtering.",
            "The basis function acts as a sieve.",
            "This process isolates pure sine wave amplitudes."
        ])

        # Assets
        magnifying_glass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifying.svg")

        # Setup Axes and Signal
        axes = Axes(x_range=[0, 2*PI, 1], y_range=[-2, 2, 1], axis_config={"include_tip": False})
        signal = FunctionGraph(lambda x: np.sin(x) + 0.5 * np.cos(2*x), x_range=[0, 2*PI], color="#FF4500")
        basis = FunctionGraph(lambda x: np.sin(x), x_range=[0, 2*PI], color="#FFFF00")
        
        # Applying VideoCritic requested layout
        self.place_in_area(axes, 'C1', 'F6', scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF4500")
        self.play(Create(axes), Create(signal))
        
        mg1 = self.place_at_grid(magnifying_glass.copy(), "C5", scale_factor=0.3)
        self.play(FadeIn(mg1))
        self.play(Create(basis.set_color("#FF4500")))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#00FFFF")
        
        area = axes.get_area(signal, x_range=[0, 2*PI], color="#00FFFF", opacity=0.3)
        self.play(Create(area))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FFFFFF")
        
        # Applying VideoCritic requested layout
        coeff_text = MathTex("a_n = \\frac{1}{\\pi} \\int f(x) \\sin(nx) dx")
        self.place_in_area(coeff_text, 'B1', 'B6', scale_factor=0.8)
        self.play(Write(coeff_text))
        
        val = DecimalNumber(0.75, num_decimal_places=2, color="#FFFFFF")
        # Applying VideoCritic requested layout
        self.place_at_grid(val, 'D4', scale_factor=0.9)
        self.play(FadeIn(val))

        mg2 = self.place_at_grid(magnifying_glass.copy(), "E5", scale_factor=0.3)
        mg2.set_color("#FF1493")
        self.play(FadeIn(mg2))
        
        self.wait(2)
