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
        # Setup layout with title and lecture lines
        self.setup_layout(
            "The Core Rules: The Axiom Checklist",
            [
                "Closure means adding two members stays in the set.",
                "Commutativity ensures the order of addition never matters.",
                "Every vector space must contain a unique zero element.",
                "These rules apply to arrows and functions alike.",
                "Axioms provide a universal language for different objects."
            ]
        )
        
        # Define colors for each stage
        COLOR_CLOSURE = "#00FF00"
        COLOR_COMM = "#FFFF00"
        COLOR_ZERO = "#FFFFFF"
        COLOR_APPLY = "#FF00FF"
        COLOR_UNIVERSAL = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Highlight lecture line
        self.play(self.lecture[0].animate.set_color(COLOR_CLOSURE))
        
        # Arrows (Left side of grid)
        arrow_origin = self.grid["C2"]
        vec_u = Arrow(arrow_origin, self.grid["B1"], buff=0, color=COLOR_CLOSURE)
        vec_v = Arrow(arrow_origin, self.grid["B3"], buff=0, color=COLOR_CLOSURE)
        vec_sum = Arrow(arrow_origin, self.grid["A2"], buff=0, color=WHITE)
        label_closure_arrow = Text("CLOSURE", font_size=18, color=COLOR_CLOSURE)
        self.place_in_area(label_closure_arrow, "A1", "A3")
        
        # Functions (Right side of grid)
        axes = Axes(x_range=[0, 1], y_range=[0, 1], x_length=2.5, y_length=2.0, tips=False).scale(0.8)
        self.place_in_area(axes, "B4", "C6")
        f_graph = axes.plot(lambda x: 0.3 * np.sin(PI * x) + 0.2, color=COLOR_CLOSURE)
        g_graph = axes.plot(lambda x: 0.2 * np.cos(2 * PI * x) + 0.1, color=COLOR_CLOSURE)
        sum_graph = axes.plot(lambda x: 0.3 * np.sin(PI * x) + 0.2 * np.cos(2 * PI * x) + 0.3, color=WHITE)
        label_closure_func = Text("CLOSURE", font_size=18, color=COLOR_CLOSURE)
        self.place_in_area(label_closure_func, "A4", "A6")

        self.play(Create(vec_u), Create(vec_v), Create(axes), Create(f_graph), Create(g_graph), run_time=1.5)
        self.play(Create(vec_sum), Create(sum_graph), Write(label_closure_arrow), Write(label_closure_func), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(COLOR_COMM))
        
        comm_arrow = Text("u + v = v + u", font_size=16, color=COLOR_COMM)
        self.place_at_grid(comm_arrow, "D2")
        comm_func = Text("f + g = g + f", font_size=16, color=COLOR_COMM)
        self.place_at_grid(comm_func, "D5")
        
        self.play(Write(comm_arrow), Write(comm_func))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(COLOR_ZERO))
        
        zero_dot = Dot(arrow_origin, color=COLOR_ZERO, radius=0.08)
        zero_label_arrow = Text("Zero Vector", font_size=16, color=COLOR_ZERO)
        self.place_at_grid(zero_label_arrow, "E2")
        
        zero_line = axes.plot(lambda x: 0, color=COLOR_ZERO)
        zero_label_func = Text("Zero Function", font_size=16, color=COLOR_ZERO)
        self.place_at_grid(zero_label_func, "E5")
        
        self.play(FadeIn(zero_dot), Write(zero_label_arrow), Create(zero_line), Write(zero_label_func))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(COLOR_APPLY))
        
        box_left = SurroundingRectangle(VGroup(vec_u, vec_v, vec_sum, label_closure_arrow), color=COLOR_APPLY)
        box_right = SurroundingRectangle(VGroup(axes, f_graph, g_graph, sum_graph, label_closure_func), color=COLOR_APPLY)
        
        self.play(Create(box_left), Create(box_right))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(COLOR_UNIVERSAL))
        
        universal_text = Text("UNIVERSAL LANGUAGE", font_size=24, color=COLOR_UNIVERSAL)
        self.place_in_area(universal_text, "F1", "F6", scale_factor=0.6)
        
        # Load asset
        waves_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/waves.svg"
        wave_l = SVGMobject(waves_path, color=COLOR_UNIVERSAL)
        wave_r = SVGMobject(waves_path, color=COLOR_UNIVERSAL)
        
        self.place_at_grid(wave_l, "F1", scale_factor=0.5)
        self.place_at_grid(wave_r, "F6", scale_factor=0.5)
        
        self.play(Write(universal_text), FadeIn(wave_l), FadeIn(wave_r))
        self.wait(2)
