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
        self.setup_layout("Newton’s Method: The Iterative Search", [
            "Newton's formula iteratively improves our guess.",
            "We calculate the next point using the derivative.",
            "Repeating this refines the root estimate rapidly."
        ])
        
        # Define the function and axis
        axes = Axes(x_range=[-1, 5], y_range=[-1, 5], axis_config={"include_tip": False})
        self.place_in_area(axes, "B3", "E6", scale_factor=0.5)
        
        func = lambda x: (x - 2)**2 + 0.5
        graph = axes.plot(func, color=BLUE)
        self.add(axes, graph)
        
        x0 = 4.0
        dot_x0 = Dot(axes.c2p(x0, func(x0)), color=YELLOW)
        label_x0 = MathTex("x_0").scale(0.7)
        self.place_at_grid(label_x0, 'C5')

        # Load assets
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg").scale(0.3)
        pencil = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pencil.svg").scale(0.3)
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg").scale(0.3)
        
        # === Animation for Lecture Line 1 ===
        self.place_at_grid(ruler, 'B1')
        self.play(FadeIn(dot_x0, label_x0, ruler))
        self.lecture[0].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Tangent line at x0
        slope = 2 * (x0 - 2)
        tangent = Line(
            axes.c2p(x0 - 1, func(x0) - slope), 
            axes.c2p(x0 + 1, func(x0) + slope), 
            color=RED
        )
        self.place_at_grid(pencil, 'B2')
        self.play(Create(tangent), FadeIn(pencil))
        self.lecture[1].set_color(RED)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Jump to x1
        x1 = x0 - func(x0) / slope
        dot_x1 = Dot(axes.c2p(x1, 0), color=GREEN)
        label_x1 = MathTex("x_1").scale(0.7)
        self.place_at_grid(label_x1, 'D3')
        
        self.place_at_grid(compass, 'F3')
        self.play(
            FadeOut(tangent, pencil),
            FadeIn(dot_x1, label_x1, compass)
        )
        self.lecture[2].set_color(GREEN)
        self.wait(2)
