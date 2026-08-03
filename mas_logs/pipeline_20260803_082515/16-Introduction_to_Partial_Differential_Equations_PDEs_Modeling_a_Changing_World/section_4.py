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
        lecture_lines = [
            "Heat moves from hot to cold.",
            "Temperature changes based on surface curvature.",
            "Sharp peaks lose heat very quickly.",
            "Valleys warm up as heat flows in.",
            "The entire distribution eventually becomes smooth."
        ]
        self.setup_layout("The Classic Example: The Heat Equation", lecture_lines)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF4D4D")
        
        # Heat Equation: du/dt = alpha * d^2u/dx^2
        equation = MathTex(
            r"\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}",
            color="#FF4D4D", font_size=40
        )
        # Resolved Issue 29: Position and scale equation for better spacing
        self.place_in_area(equation, 'A2', 'A5', scale_factor=0.8)
        
        # Visualizing the rod
        rod = Rectangle(height=0.3, width=5, fill_opacity=1).set_stroke(WHITE, 1)
        rod.set_fill(color=[RED, BLUE, RED]) # Red ends, blue middle for peak visualization
        self.place_in_area(rod, "F1", "F6")
        
        self.play(Write(equation))
        self.play(Create(rod))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#4DFF4D")
        
        # Axes for the plot (within grid B1-E6)
        axes_center = self.grid["D3"] + RIGHT * 0.5
        axes = Axes(
            x_range=[-2.5, 2.5, 1],
            y_range=[0, 1.5, 0.5],
            x_length=5,
            y_length=3,
            axis_config={"include_tip": False, "color": GRAY}
        ).move_to(axes_center)
        
        labels = axes.get_axis_labels(x_label="x", y_label="u(x)")
        
        # Temperature distribution function u(x, t)
        # Using a Gaussian that flattens over time
        # Start with small sigma for sharp peak
        time_tracker = ValueTracker(0.3) 
        
        # Initial graph
        graph = axes.plot(
            lambda x: np.exp(-(x**2) / (2 * 0.3**2)),
            color="#4DFF4D"
        )
        
        # Persistent updater for the graph shape
        def update_graph(m):
            t = time_tracker.get_value()
            m.become(axes.plot(
                lambda x: np.exp(-(x**2) / (2 * t**2)),
                color="#4DFF4D"
            ))

        self.play(Create(axes), Create(labels))
        self.play(Create(graph))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FFFF4D")
        
        # Circle the peak (position updated by updater if needed, but here it's static)
        peak_circle = Circle(radius=0.3, color="#FFFF4D")
        peak_circle.move_to(axes.c2p(0, 1))
        
        peak_label = Text("High Curvature", font_size=18, color="#FFFF4D")
        # Resolved Issue 28: Centered alignment at B3-B4
        self.place_in_area(peak_label, 'B3', 'B4', scale_factor=0.8)
        
        self.play(Create(peak_circle), Write(peak_label))
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color("#4DFFFF")
        
        # Indicators for valleys (edges in this Gaussian model)
        valley_arrow_l = Arrow(start=axes.c2p(-2, 0.5), end=axes.c2p(-1.5, 0.2), color="#4DFFFF", buff=0)
        valley_arrow_r = Arrow(start=axes.c2p(2, 0.5), end=axes.c2p(1.5, 0.2), color="#4DFFFF", buff=0)
        
        self.play(GrowArrow(valley_arrow_l), GrowArrow(valley_arrow_r))
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color("#FF4DFF")
        
        # Animate the flattening
        self.play(FadeOut(peak_circle), FadeOut(peak_label), FadeOut(valley_arrow_l), FadeOut(valley_arrow_r))
        
        # Update rod color to match flattening distribution
        new_rod_fill = rod.copy().set_fill(color=[BLUE_D, BLUE_B, BLUE_D])
        
        # Activate graph updater for the animation
        graph.add_updater(update_graph)
        
        self.play(
            time_tracker.animate.set_value(1.5),
            Transform(rod, new_rod_fill),
            run_time=5,
            rate_func=linear
        )
        
        # Clean up updater
        graph.remove_updater(update_graph)
        
        self.wait(3)
