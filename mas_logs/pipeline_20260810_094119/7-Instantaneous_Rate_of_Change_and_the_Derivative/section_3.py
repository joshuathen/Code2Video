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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Derivative is the limit of the secant.", 
            "Definition uses h approaching zero.", 
            "It calculates slope at a single point.", 
            "For x squared, the slope is 4.", 
            "This is the instantaneous speed."
        ]
        self.setup_layout("Defining the Derivative", lecture_lines)
        
        # Formula based on instruction for readability/centering
        formula = MathTex(
            "f'(x) = \\lim_{h \\to 0} \\frac{f(x+h) - f(x)}{h}",
            font_size=40
        )
        self.place_in_area(formula, 'B3', 'D6', scale_factor=0.75)

        # Asset for animation 5
        icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")

        # === Animation for Lecture Line 1 ===
        self.play(Write(formula))
        self.lecture[0].set_color(WHITE)

        # === Animation for Lecture Line 2 ===
        # "[Asset: formal_definition_formula] Definition uses h approaching zero."
        self.lecture[1].set_color("#FF8C00")
        self.play(Indicate(formula, color="#FF8C00"))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FF00")
        axes = Axes(x_range=[-1, 4, 1], y_range=[-1, 5, 1], axis_config={"include_tip": False})
        curve = axes.plot(lambda x: x**2, color=YELLOW)
        graph = VGroup(axes, curve)
        self.place_in_area(graph, 'D1', 'E6', scale_factor=0.4)
        self.play(Create(graph))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FF4500")
        slope_val = Text("4", color="#FF4500", font_size=36)
        self.place_at_grid(slope_val, 'F3')
        self.play(Write(slope_val))

        # === Animation for Lecture Line 5 ===
        # "[Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg]"
        self.lecture[4].set_color("#00FFFF")
        self.place_at_grid(icon, 'F5', scale_factor=1.0)
        self.play(
            FadeIn(icon),
            slope_val.animate.set_color("#00FFFF").scale(1.2)
        )
        self.wait(1)
