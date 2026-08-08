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
        self.setup_layout("Prerequisite Review: Rates of Change", [
            "A derivative is a function's instantaneous slope.",
            "Think of it as velocity from position.",
            "Watch the tangent line slide along the curve."
        ])

        # Assets
        car = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/car.png")
        odometer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/odometer.svg")
        
        # Plot Setup
        axes = Axes(x_range=[-2, 2], y_range=[-1, 3], x_length=3, y_length=2)
        curve = axes.plot(lambda x: x**2 + 0.5, color=WHITE)
        curve_group = VGroup(axes, curve)
        
        # Positioning curve
        self.place_in_area(curve_group, 'A3', 'C6', scale_factor=0.7)
        
        # Positioning car asset
        self.place_at_grid(car, 'A2', scale_factor=0.3)

        # Dynamic objects
        point = ValueTracker(-1.2)
        dot = Dot(color=RED)
        dot.add_updater(lambda d: d.move_to(axes.c2p(point.get_value(), point.get_value()**2 + 0.5)))
        
        tangent = TangentLine(curve, alpha=0.2, length=1.0, color=RED)
        tangent.add_updater(lambda t: t.become(TangentLine(curve, alpha=(point.get_value()+2)/4, length=1.0, color=RED)))
        
        slope_label = MathTex(f"f'(x)", color=GREEN).scale(0.8)
        self.place_at_grid(slope_label, 'D4', scale_factor=0.8)
        
        self.place_at_grid(odometer, 'D5', scale_factor=0.4)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFF00")
        self.play(Create(axes), Create(curve), FadeIn(car))
        self.add(dot, tangent)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FFFF")
        self.play(point.animate.set_value(1.2), run_time=3, rate_func=linear)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FF00")
        self.play(FadeIn(slope_label), FadeIn(odometer))
        self.wait(1)
