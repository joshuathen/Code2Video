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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Synthesis: Where Geometry Meets Arithmetic", [
            "Prime density relates to logarithmic Pi approximations.",
            "Local randomness follows global mathematical laws.",
            "Geometric and arithmetic worlds are deeply connected."
        ])
        
        # Asset loader (simplified as Manim treats SVGs as SVGMobject)
        grid_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        axes = Axes(x_range=[0, 10, 1], y_range=[0, 3, 1], axis_config={"include_tip": False})
        curve = axes.plot(lambda x: np.log(x + 1), color=BLUE)
        primes = [2, 3, 5, 7]
        dots = VGroup(*[Dot(axes.c2p(p, 0), color=RED) for p in primes])
        
        # Group with asset
        vis = VGroup(axes, curve, dots, grid_asset.copy())
        self.place_in_area(vis, 'B4', 'F6', scale_factor=0.5)
        self.play(Create(axes), Create(curve), FadeIn(dots), FadeIn(grid_asset))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(GREEN)
        circle = Circle(radius=1, color=WHITE)
        self.place_at_grid(circle, 'E2', scale_factor=0.4)
        
        # Add text (SVM example per feedback)
        svm_text = Text("Global Law", font_size=18, color=GREEN)
        self.place_in_area(svm_text, 'D1', 'F2', scale_factor=0.6)
        
        self.play(Create(circle), Write(svm_text))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(PURPLE)
        line = Line(vis.get_center(), circle.get_center(), color=ORANGE)
        self.play(Create(line))
        self.wait(2)
