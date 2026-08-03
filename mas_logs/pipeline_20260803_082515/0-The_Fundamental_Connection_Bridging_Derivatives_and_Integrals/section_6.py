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

class Section6Scene(TeachingScene):
    def construct(self):
        self.setup_layout(
            "Real-World Application: The Water Tank",
            [
                "Imagine water flowing into a tank at rate f(t).",
                "The flow rate is the derivative of volume.",
                "Integrating this rate gives the total water volume."
            ]
        )

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Pipe asset
        pipe = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pipe.svg")
        pipe.set_color("#A9A9A9")
        self.place_at_grid(pipe, "A2", scale_factor=0.6)
        
        # Bucket asset
        bucket = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bucket.svg")
        bucket.set_color("#FFFFFF")
        self.place_at_grid(bucket, "C2", scale_factor=0.8)
        
        # Water in Bucket (Persistent Mobject)
        water_level = ValueTracker(0.01)
        water_color = "#0000FF"
        
        # We'll use a Rectangle for water and update its height
        water = Rectangle(width=0.5, height=0.01, color=water_color, fill_color=water_color, fill_opacity=0.7, stroke_width=0)
        water.move_to(bucket.get_bottom() + UP * 0.1, aligned_edge=DOWN)
        
        water.add_updater(lambda m: m.stretch_to_fit_height(max(0.01, water_level.get_value()), about_edge=DOWN))
        
        # Water drops
        drops = VGroup(*[Dot(radius=0.04, color=water_color) for _ in range(3)])
        for i, d in enumerate(drops):
            d.move_to(pipe.get_center() + DOWN * (0.2 + i * 0.4))

        def update_drops(group, dt):
            for d in group:
                d.shift(DOWN * 3 * dt)
                # If drop hits bucket or goes too low
                if d.get_center()[1] < bucket.get_center()[1] + 0.3:
                    d.move_to(pipe.get_center() + DOWN * 0.1)

        self.add(pipe, bucket, water)
        self.play(FadeIn(pipe), FadeIn(bucket))
        
        drops.add_updater(update_drops)
        self.add(drops)
        
        self.play(
            water_level.animate.set_value(1.0),
            rate_func=linear,
            run_time=4
        )
        
        drops.remove_updater(update_drops)
        self.remove(drops)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(GREEN)
        
        # Labels for association (Issue 35)
        flow_label = MathTex("f(t)", color=YELLOW)
        vol_label = MathTex("V(t)", color=BLUE)
        self.place_at_grid(flow_label, 'A3', scale_factor=0.8)
        self.place_at_grid(vol_label, 'C3', scale_factor=0.8)
        
        # Derivative relation (Issue 36)
        deriv_relation = MathTex("f(t) = \\frac{dV}{dt}", color=GREEN)
        self.place_at_grid(deriv_relation, 'B4', scale_factor=0.8)
        
        self.play(Write(flow_label), Write(vol_label))
        self.play(Write(deriv_relation))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(ORANGE)
        
        # Plot axes in D-F area
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 3, 1],
            x_length=3,
            y_length=2,
            axis_config={"include_tip": True}
        ).set_color(GRAY)
        self.place_in_area(axes, "D2", "F5", scale_factor=0.9)
        
        rate_curve_color = "#FF4500"
        # f(t) = 0.5 * sin(t) + 1.5
        curve = axes.plot(lambda x: 0.5 * np.sin(x) + 1.5, x_range=[0, 4], color=rate_curve_color)
        curve_label = MathTex("f(t)", color=rate_curve_color).scale(0.6)
        curve_label.next_to(curve, UP, buff=0.1)
        
        # Area under curve
        area = axes.get_area(curve, x_range=[0, 3], color=water_color, opacity=0.3)
        
        # Integral formula (Issue 37)
        integral_formula = MathTex("V(T) = \\int_{0}^{T} f(t) dt", color=ORANGE)
        self.place_in_area(integral_formula, 'F3', 'F5', scale_factor=0.8)
        
        self.play(Create(axes), Create(curve), Write(curve_label))
        self.play(FadeIn(area), Write(integral_formula))
        
        self.wait(2)
