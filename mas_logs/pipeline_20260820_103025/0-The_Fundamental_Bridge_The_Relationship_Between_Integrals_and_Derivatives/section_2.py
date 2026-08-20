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
        self.setup_layout("The Geometric Connection", ["Tangent slope defines the derivative.", "Area under curves defines the integral.", "Small rectangles sum to the total area."])
        
        # Setup Assets
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        calculator = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg")
        
        # Setup Axes
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 4, 1], axis_config={"include_tip": False})
        self.place_in_area(axes, "C2", "F5", scale_factor=0.55)
        
        curve = axes.plot(lambda x: 0.2*x**2 + 1, x_range=[0.5, 3.5], color=WHITE)
        
        # === Animation for Lecture Line 1 ===
        # Tangent slope defines the derivative.
        self.lecture[0].set_color("#FFFFFF")
        self.place_at_grid(ruler, "A4", scale_factor=0.3)
        self.play(FadeIn(axes), Create(curve), FadeIn(ruler))
        
        # === Animation for Lecture Line 2 ===
        # Area under curves defines the integral.
        self.lecture[1].set_color("#00FFFF")
        area = axes.get_area(curve, x_range=[0.5, 3.5], color="#00FFFF", opacity=0.3)
        self.play(FadeIn(area))
        
        # === Animation for Lecture Line 3 ===
        # Small rectangles sum to the total area.
        self.lecture[2].set_color("#FFCC00")
        
        rects = axes.get_riemann_rectangles(curve, x_range=[0.5, 3.5], dx=0.3, color="#FFCC00", fill_opacity=0.6)
        
        self.place_at_grid(calculator, "A6", scale_factor=0.3)
        self.play(FadeIn(rects), FadeIn(calculator))
        self.wait(2)
