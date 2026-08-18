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
            "Area rate equals current curve height.",
            "Adding a slice increases total area.",
            "Integration and differentiation are inverses.",
            "Derivative of area is original function.",
            "Area changes match the curve's height."
        ]
        self.setup_layout("The Core Connection: Differentiation and Integration", lecture_lines)
        
        # Define objects
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 3, 1], axis_config={"include_tip": False})
        curve = axes.plot(lambda x: 0.2*x**2 + 0.5, color=WHITE)
        
        # Assets
        knife_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/knife.svg")
        slicer_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/slicer.svg")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#E67E22")
        self.place_in_area(axes, "A1", "C4", scale_factor=0.6)
        self.add(curve)
        dot = Dot(curve.point_from_proportion(0.6), color="#E67E22")
        self.place_at_grid(knife_icon, "A5", scale_factor=0.3)
        self.add(dot, knife_icon)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#3498DB")
        slice_rect = axes.get_riemann_rectangles(curve, x_range=[2, 2.3], dx=0.3, color="#3498DB")
        self.play(Create(slice_rect))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#F1C40F")
        arrow = Arrow(start=self.grid["B5"], end=self.grid["D5"], color="#F1C40F")
        self.play(Create(arrow))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#E74C3C")
        derivative_text = MathTex(r"A'(x) = f(x)", color="#E74C3C")
        self.place_at_grid(derivative_text, "D5", scale_factor=0.9)
        self.play(Write(derivative_text))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#2ECC71")
        connection_line = Line(dot.get_center(), slice_rect.get_center(), color="#2ECC71")
        self.place_at_grid(slicer_icon, "E5", scale_factor=0.3)
        self.play(Create(connection_line), FadeIn(slicer_icon))
        self.wait(2)
