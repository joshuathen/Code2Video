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
        lecture_lines = [
            "Richardson proposed energy cascades through scales.",
            "Kolmogorov defined large-scale energy entry.",
            "Energy dissipates at the smallest scales.",
            "Spectrum E(k) follows power law slope.",
            "Scaling exponent equals negative five-thirds."
        ]
        self.setup_layout("The Kolmogorov Cascade: Mathematical Scaling", lecture_lines)
        
        # Elements
        cascade = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cascade.svg")
        turbine = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/turbine.svg")
        whirlpool = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/whirlpool.svg")
        
        # Pre-positioning
        self.place_at_grid(cascade, 'B5', scale_factor=0.6)
        self.place_at_grid(turbine, 'B5', scale_factor=0.6)
        self.place_at_grid(whirlpool, 'B5', scale_factor=0.6)
        
        axes = Axes(x_range=[0, 5, 1], y_range=[0, 5, 1], axis_config={"include_tip": False})
        graph = axes.plot(lambda x: x**(-5/3) * 5 if x > 0 else 0, x_range=[0.5, 4])
        
        self.place_in_area(axes, 'C3', 'E5', scale_factor=0.7)
        self.place_in_area(graph, 'C3', 'E5', scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.play(FadeIn(cascade))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        self.play(ReplacementTransform(cascade, turbine.set_color("#FFFF00")))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        self.play(turbine.animate.scale(0.5))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FF00FF"))
        self.play(FadeIn(axes), FadeIn(graph.set_color("#FF00FF")))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FF00FF"))
        self.play(ReplacementTransform(turbine, whirlpool.set_color("#FF00FF")))
        self.wait(2)
