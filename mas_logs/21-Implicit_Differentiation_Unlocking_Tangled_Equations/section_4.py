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
        # 1. Setup Layout & Lecture Lines
        lecture_lines = [
            "The circle equation links x and y together.",
            "Differentiate x-squared and y-squared separately.",
            "Move the x-terms to the other side.",
            "Divide to isolate the slope dy/dx.",
            "Plugging in coordinates gives the tangent's exact slope."
        ]
        self.setup_layout("Graphical Deep Dive: The Circle's Slope", lecture_lines)

        # 2. Graph Group elements
        axes = Axes(
            x_range=[-6, 6, 2],
            y_range=[-6, 6, 2],
            x_length=5,
            y_length=5,
            axis_config={"include_tip": True}
        )
        circle = axes.plot_implicit_curve(
            lambda x, y: x**2 + y**2 - 25,
            color=BLUE
        )
        dot = Dot(axes.c2p(3, 4), color=RED)
        # Tangent line: y - 4 = -0.75(x - 3) => y = -0.75x + 6.25
        tangent_line = axes.plot(
            lambda x: -0.75 * x + 6.25,
            x_range=[-1, 6],
            color=YELLOW
        )
        
        graph_group = VGroup(axes, circle, dot, tangent_line)
        self.place_in_area(graph_group, 'A3', 'F6', scale_factor=0.9)
        
        coord_label = Text("(3, 4)", font_size=24)
        self.place_at_grid(coord_label, 'B5', scale_factor=0.6)

        # 3. Formula Group elements
        eq1 = Text("x² + y² = 25", font_size=32)
        
        # eq2: 2x + 2y(dy/dx) = 0
        eq2_part1 = Text("2x")
        eq2_part2 = Text(" + ")
        eq2_part3 = Text("2y(dy/dx)", color="#FF0000")
        eq2_part4 = Text(" = 0")
        eq2 = VGroup(eq2_part1, eq2_part2, eq2_part3, eq2_part4).arrange(RIGHT, buff=0.1)
        
        # eq3: 2y(dy/dx) = -2x
        eq3_p1 = Text("2y(dy/dx)", color="#FF0000")
        eq3_p2 = Text(" = -2x")
        eq3 = VGroup(eq3_p1, eq3_p2).arrange(RIGHT, buff=0.1)
        
        # eq4: dy/dx = -x/y (Boxed)
        eq4_base = Text("dy/dx = -x/y", color=GREEN)
        box = SurroundingRectangle(eq4_base, color="#00FF00", buff=0.1)
        eq4 = VGroup(eq4_base, box)
        
        # eq5: m = -3/4
        eq5 = Text("m = -3/4", color=YELLOW)

        # Build combined formula group for layout
        formula_group = VGroup(eq1, eq2, eq3, eq4, eq5).arrange(DOWN, buff=0.6, aligned_edge=LEFT)
        self.place_in_area(formula_group, 'A1', 'F2', scale_factor=0.7)

        # Hide elements for sequential animation
        eq2.set_opacity(0)
        eq3.set_opacity(0)
        eq4.set_opacity(0)
        eq5.set_opacity(0)
        tangent_line.set_opacity(0)
        dot.set_opacity(0)
        coord_label.set_opacity(0)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(Create(axes), Create(circle), Write(eq1), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.play(eq2.animate.set_opacity(1), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        # Simulate "2x" sliding and becoming "-2x"
        self.play(
            FadeOut(eq2_part1, shift=RIGHT),
            FadeOut(eq2_part2),
            FadeOut(eq2_part4),
            ReplacementTransform(eq2_part3, eq3_p1),
            FadeIn(eq3_p2),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        self.play(eq4.animate.set_opacity(1), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        self.play(
            FadeIn(dot, scale=0.5),
            FadeIn(coord_label),
            Write(eq5),
            Create(tangent_line),
            run_time=2
        )
        self.wait(2)
