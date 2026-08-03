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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("A Non-Geometric World: Function Spaces", [
            "Functions can be treated exactly like geometric vectors.",
            "Adding two functions produces another valid function.",
            "This proves that \"vectors\" can be wavy curves."
        ])
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Create axes on the right side
        axes = Axes(
            x_range=[-2.2, 2.2, 1],
            y_range=[-1.2, 4.2, 1],
            axis_config={"include_tip": False, "color": GREY_B},
            x_length=4.5,
            y_length=4.5
        )
        self.place_in_area(axes, "A1", "F6")
        
        f_graph = axes.plot(lambda x: x**2, x_range=[-2, 2], color="#00FFFF")
        g_graph = axes.plot(lambda x: x, x_range=[-2, 2], color="#FFA500")
        
        f_label = MathTex("f(x)=x^2", color="#00FFFF", font_size=24)
        g_label = MathTex("g(x)=x", color="#FFA500", font_size=24)
        
        self.place_at_grid(f_label, "B2", scale_factor=0.8)
        # Resolved Issue 39: Move g_label from D2 to E1 to avoid overlap
        self.place_at_grid(g_label, "E1", scale_factor=0.8)

        self.play(Write(axes))
        self.play(Create(f_graph), Create(g_graph))
        self.play(Write(f_label), Write(g_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        h_graph = axes.plot(lambda x: x**2 + x, x_range=[-2, 1.5], color="#FFFFFF")
        h_label = MathTex("h(x)=f(x)+g(x)", color="#FFFFFF", font_size=24)
        # Resolved Issue 40: Move h_label from A4 to B4 to avoid title occlusion
        self.place_at_grid(h_label, "B4", scale_factor=0.8)
        
        # Show some vertical bars to visualize point-wise addition
        sample_x = [-1.5, -0.5, 0.5, 1.2]
        addition_lines = VGroup()
        for x in sample_x:
            p_f = axes.c2p(x, x**2)
            p_h = axes.c2p(x, x**2 + x)
            # A line starting from f(x) and going up/down by g(x)
            line = Line(p_f, p_h, color="#FFA500", stroke_width=3)
            dot = Dot(p_h, radius=0.05, color=WHITE)
            addition_lines.add(VGroup(line, dot))

        self.play(Create(addition_lines))
        self.play(Create(h_graph))
        self.play(Write(h_label))
        self.wait(2)
        self.play(FadeOut(addition_lines))

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Transition: show scaling of f(x)
        f_scaled_graph = axes.plot(lambda x: 0.5 * x**2, x_range=[-2, 2], color="#00FFFF")
        f_scaled_label = MathTex("0.5 \\cdot f(x)", color="#00FFFF", font_size=24)
        # Resolved Issue 38: Move f_scaled_label from C6 to C5 to avoid frame edge
        self.place_at_grid(f_scaled_label, "C5", scale_factor=0.8)
        
        self.play(
            FadeOut(g_graph, g_label, h_graph, h_label),
            f_graph.animate.set_stroke(opacity=0.3)
        )
        self.play(Transform(f_graph.copy(), f_scaled_graph), run_time=2)
        self.play(Write(f_scaled_label))
        self.wait(3)
