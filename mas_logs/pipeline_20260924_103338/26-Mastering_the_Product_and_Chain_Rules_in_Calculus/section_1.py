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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout(
            "Prerequisite Warm-up: The Concept of Rate",
            ["Derivatives measure instantaneous change as a rate.", "Think of functions as input-output machines.", "Complex functions nest or multiply smaller ones."]
        )
        
        # === Animation for Lecture Line 1 ===
        # Derivatives measure instantaneous change as a rate.
        # Visualize tangent line slope (using #00FF00)
        self.lecture[0].set_color("#00FF00")
        
        curve = FunctionGraph(lambda x: 0.1 * x**3, x_range=[-2, 2], color=WHITE)
        self.place_at_grid(curve, "B2", scale_factor=0.8)
        
        dot = Dot(color="#00FF00")
        dot.move_to(curve.point_from_proportion(0.5))
        
        tangent = always_redraw(lambda: TangentLine(curve, alpha=0.5, length=1.5, color="#00FF00"))
        
        self.add(curve, dot, tangent)
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Think of functions as input-output machines.
        self.lecture[1].set_color("#FFFFFF")
        
        # Load asset machine icon
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/machine.svg]
        machine = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/machine.svg")
        self.place_at_grid(machine, "E3", scale_factor=1.0)
        
        x_label = MathTex("x", color=WHITE).next_to(machine, LEFT)
        f_x_label = MathTex("f(x)", color=WHITE).next_to(machine, RIGHT)
        
        self.play(Write(machine), Write(x_label), Write(f_x_label))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Complex functions nest or multiply smaller ones.
        self.lecture[2].set_color("#FF00FF")
        
        func1 = MathTex("f(g(x))", color="#FF00FF").scale(1.5)
        func2 = MathTex("f(x) \\cdot g(x)", color="#FF00FF").scale(1.5)
        
        self.place_at_grid(func1, "C5", scale_factor=1.0)
        self.play(FadeIn(func1))
        self.wait(1)
        self.play(Transform(func1, func2))
        self.wait(2)
