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

class Section2DerivativeScene(TeachingScene):
    def construct(self):
        # 1. Setup Layout
        self.setup_layout(
            "The Derivative: Zooming into 'The Now'",
            [
                "A derivative represents the instantaneous rate of change.",
                "Zoom into a curve until it looks straight.",
                "This straight line's slope is the derivative f'(x).",
                "Like a speedometer showing speed at one moment.",
                "It captures the precise \"steepness of the now.\""
            ]
        )

        # Colors
        COLOR_CURVE = "#00BFFF"
        COLOR_SPEEDO = "#FF4500"
        COLOR_TANGENT = "#FFFF00"
        COLOR_SLOPE = "#FFD700"
        COLOR_TEXT = "#FFFFFF"

        # Asset Paths
        CAR_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/car.png"

        # === Animation for Lecture Line 1 ===
        # A derivative represents the instantaneous rate of change.
        self.lecture[0].set_color(COLOR_CURVE)
        
        # Create Axes and Curve
        # s(t) = 12 * t^2. s'(t) = 24t. At t=2.5, speed=60.
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 300, 100],
            x_length=4.5,
            y_length=4,
            axis_config={"include_tip": True, "color": GREY}
        )
        self.place_in_area(axes, "B1", "F6")
        
        curve = axes.plot(lambda x: 12 * x**2, x_range=[0, 4.5], color=COLOR_CURVE)
        curve_label = Text("s(t) = 12t^2", font_size=20, color=COLOR_CURVE)
        # Issue 24 Fix
        self.place_in_area(curve_label, 'B2', 'B3', scale_factor=0.8)

        # Race car asset integration
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/car.png]
        car = ImageMobject(CAR_ASSET).scale(0.3)
        
        # Position car on curve
        t_tracker = ValueTracker(0)
        car.add_updater(lambda m: m.move_to(axes.c2p(t_tracker.get_value(), 12 * t_tracker.get_value()**2)))

        self.add(axes, curve, curve_label, car)
        self.play(t_tracker.animate.set_value(4), run_time=3, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Zoom into a curve until it looks straight.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_TANGENT)
        
        t_focus = 2.5
        # Zoomed view: focus on t=2.5. Range [2.4, 2.6].
        zoomed_axes = Axes(
            x_range=[2.4, 2.6, 0.1],
            y_range=[65, 85, 10],
            x_length=4.5,
            y_length=4,
            axis_config={"include_tip": False, "color": GREY_B}
        )
        self.place_in_area(zoomed_axes, "B1", "F6")
        
        zoomed_curve = zoomed_axes.plot(lambda x: 12 * x**2, x_range=[2.4, 2.6], color=COLOR_CURVE)
        dot_focus = Dot(zoomed_axes.c2p(t_focus, 12 * t_focus**2), color=WHITE)
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/car.png] (fading out car here)
        self.play(
            FadeOut(axes), FadeOut(curve), FadeOut(curve_label), FadeOut(car),
            FadeIn(zoomed_axes), FadeIn(zoomed_curve), FadeIn(dot_focus)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # This straight line's slope is the derivative f'(x).
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_SLOPE)
        
        # Correctly calling TangentLine for the zoomed curve
        tangent_line = TangentLine(zoomed_curve, alpha=0.5, length=6, color=COLOR_TANGENT)
        
        slope_label = Text("f'(2.5) = 60", font_size=24, color=COLOR_SLOPE)
        # Issue 25 Fix
        self.place_in_area(slope_label, 'B5', 'B6', scale_factor=0.8)

        self.play(Create(tangent_line))
        self.play(Write(slope_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Like a speedometer showing speed at one moment.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(COLOR_SPEEDO)
        
        # Speedometer visual
        speedo_arc = Arc(radius=0.7, start_angle=PI, angle=-PI, color=GREY).set_stroke(width=4)
        speedo_center = self.grid["D2"]
        speedo_arc.move_to(speedo_center + UP*0.3)
        
        # Needle points up (60 is midpoint)
        needle = Line(speedo_arc.get_center(), speedo_arc.get_center() + UP*0.6, color=COLOR_SPEEDO, stroke_width=4)
        
        speedo_val = Text("60 km/h", font_size=18, color=COLOR_SPEEDO)
        speedo_val.next_to(speedo_arc, DOWN, buff=0.1)
        
        speedo_group = VGroup(speedo_arc, needle, speedo_val)
        
        self.play(FadeIn(speedo_group))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # It captures the precise "steepness of the now."
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(COLOR_TEXT)
        
        # Final notation
        final_notation = Text("f'(x) = slope at x", font_size=32, color=COLOR_TEXT)
        # Issue 26 Fix
        self.place_in_area(final_notation, 'E4', 'F6', scale_factor=0.8)
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/car.png] 
        self.play(
            FadeOut(zoomed_axes), FadeOut(zoomed_curve), FadeOut(dot_focus), 
            FadeOut(tangent_line), FadeOut(slope_label), FadeOut(speedo_group)
        )
        # Restore background context
        self.play(FadeIn(axes), FadeIn(curve), Write(final_notation))
        self.wait(2)
