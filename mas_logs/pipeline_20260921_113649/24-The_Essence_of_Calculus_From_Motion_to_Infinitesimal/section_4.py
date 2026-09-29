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
        lecture_lines = [
            "Integrals sum up tiny slices.",
            "Adding infinite slices reveals total area.",
            "It acts as the derivative's opposite.",
            "Think of it as accumulating change.",
            "This bridges tiny parts to the whole."
        ]
        self.setup_layout("The Integral: Accumulating the Whole", lecture_lines)
        
        # Assets
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        graph = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/graph.svg")
        
        # Setup visual
        axes = Axes(x_range=[-2, 2, 1], y_range=[0, 1, 0.5], axis_config={"include_tip": False})
        curve = axes.plot(lambda x: 0.1 * x**3 - 0.2 * x**2 + 0.5, color=BLUE)
        
        # Group for integration visual
        integral_visual = VGroup(axes, curve)
        self.place_in_area(integral_visual, "B3", "E5", scale_factor=0.75)
        
        def get_bars(n):
            bars = VGroup()
            x_vals = np.linspace(-2, 2, n+1)
            f = lambda x: 0.1 * x**3 - 0.2 * x**2 + 0.5
            for i in range(n):
                # Riemann rectangles in Manim CE expect a graph object and x_range
                bar = axes.get_riemann_rectangles(curve, 
                                                x_range=[x_vals[i], x_vals[i+1]], dx=x_vals[i+1]-x_vals[i], color="#FFA500")
                bars.add(bar)
            return bars

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFA500"))
        self.place_at_grid(ruler, "A6", scale_factor=0.5)
        bars_4 = get_bars(4)
        self.play(FadeIn(integral_visual), Create(bars_4), FadeIn(ruler))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        self.place_at_grid(graph, "F6", scale_factor=0.5)
        bars_20 = get_bars(20)
        self.play(ReplacementTransform(bars_4, bars_20), FadeIn(graph))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FF00FF"))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#00FF00"))
        self.wait(1)
