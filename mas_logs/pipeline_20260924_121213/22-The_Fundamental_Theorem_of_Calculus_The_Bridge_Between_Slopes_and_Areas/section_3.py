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
        lines = ["The derivative of area is the height.", "This reveals the inverse relationship of calculus.", "Differentiation cancels out the accumulation.", "The area changes at rate f(x).", "Calculus connects local slopes to total area."]
        self.setup_layout("The Fundamental Theorem (Part 1)", lines)
        
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 3, 1], axis_config={"include_tip": False}).scale(0.5)
        curve = axes.plot(lambda x: 0.5 * x**2 + 1, color=BLUE)
        label = MathTex("f(x)").next_to(curve, UP)
        plot_group = VGroup(axes, curve, label)
        
        self.place_in_area(plot_group, 'C3', 'E5', scale_factor=1.2)
        self.add(plot_group)
        
        # Assets
        calculator_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg")
        graph_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/graph.svg")
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.place_at_grid(calculator_icon, 'B2', scale_factor=0.3)
        self.play(FadeIn(calculator_icon))
        formula = MathTex(r"\frac{d}{dx} \int_a^x f(t) dt = f(x)", color=WHITE)
        self.place_in_area(formula, 'B4', 'B6', scale_factor=0.9)
        self.play(Write(formula))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF00FF"))
        
        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FFFF00"))
        x_val = 2.5
        rect = axes.get_riemann_rectangles(curve, x_range=[x_val, x_val+0.2], dx=0.2, color=YELLOW, fill_opacity=0.5)
        self.play(Create(rect))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#00FFFF"))
        self.place_at_grid(graph_icon, 'E5', scale_factor=0.3)
        self.play(FadeIn(graph_icon))
        
        animation_label = Text("Accumulation Process", font_size=18, color=WHITE)
        self.place_at_grid(animation_label, 'F2', scale_factor=0.8)
        self.play(Write(animation_label))
        
        self.wait(2)
