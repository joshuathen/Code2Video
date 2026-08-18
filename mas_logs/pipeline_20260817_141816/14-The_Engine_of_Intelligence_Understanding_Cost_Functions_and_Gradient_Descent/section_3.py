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
            "Gradient descent guides the network downhill.",
            "The gradient shows the direction of steepest slope.",
            "We move in the negative direction of the gradient.",
            "Weights update incrementally with each step.",
            "Small steps help navigate toward the minimum."
        ]
        self.setup_layout("The Mechanism of Gradient Descent", lecture_lines)
        
        # Asset usage
        mountain = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/mountain.svg")
        axes = Axes(x_range=[-3, 3], y_range=[-1, 5], axis_config={"include_tip": False}).scale(0.4)
        curve = axes.plot(lambda x: 0.5 * x**2 + 1, color=BLUE)
        
        graph_group = VGroup(mountain, axes, curve)
        self.place_in_area(graph_group, 'D4', 'F6', scale_factor=0.9)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        loss_formula = MathTex("J(w) = \\frac{1}{2}w^2 + 1", color="#FFFF00")
        self.place_at_grid(loss_formula, 'C4', scale_factor=0.7)
        self.play(Write(loss_formula))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF00FF"))
        tangent = Line(start=axes.c2p(-1.5, 2.125), end=axes.c2p(-0.5, 1.125), color="#FF00FF")
        # Ensure tangent is relative to axes/mountain
        self.play(Create(tangent))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        dot = Dot(axes.c2p(-1.5, 2.125), color=RED)
        self.add(dot)
        
        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#00FF00"))
        # Move dot down along the curve/mountain
        self.play(dot.animate.move_to(axes.c2p(-0.5, 1.125)))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FFA500"))
        val_text = Text("Cost: 2.12 -> 1.12", font_size=20, color="#FFA500")
        self.place_at_grid(val_text, 'E5', scale_factor=0.7)
        self.play(Write(val_text))
        self.play(dot.animate.move_to(axes.c2p(0, 1)))
