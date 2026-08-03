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

class Section2Scene(TeachingScene):
    def construct(self):
        # Data from storyboard
        title_text = "Prerequisite Visual: Slopes and Areas"
        lecture_lines = [
            "Geometrically, derivatives represent the slope at a point.",
            "Integrals represent the area under a curve.",
            "One looks microscopically, the other macroscopically."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        DERIVATIVE_COLOR = "#FF4500"
        INTEGRAL_COLOR = "#32CD32"
        GLASS_COLOR = "#B0C4DE"
        CURVE_COLOR = "#FFFFFF"

        # Define the function explicitly to reuse it
        curve_func = lambda x: 0.15 * (x - 1)**2 + 1

        # === Animation for Lecture Line 1 ===
        # Highlight Line 1
        self.play(self.lecture[0].animate.set_color(DERIVATIVE_COLOR))
        
        # Setup Axes and Curve
        # Fixed: issue 26 - use A1 to F6
        axes = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 4, 1],
            x_length=4,
            y_length=4,
            axis_config={"include_tip": False, "color": GRAY}
        )
        self.place_in_area(axes, 'A1', 'F6', scale_factor=0.8)
        
        curve = axes.plot(curve_func, x_range=[0, 4], color=CURVE_COLOR)
        
        # Point and Tangent Line
        point_x = 2.5
        target_point = axes.c2p(point_x, curve_func(point_x))
        dot = Dot(target_point, color=DERIVATIVE_COLOR)
        
        # In Manim CE, use TangentLine class with alpha proportion (point_x / x_range_max)
        tangent_line = TangentLine(curve, alpha=point_x/4, length=4, color=DERIVATIVE_COLOR)
        
        self.play(Create(axes), Create(curve), run_time=1.5)
        self.play(Create(dot), Create(tangent_line))

        # Magnifying Glass Asset
        # Fixed: issue 20 - use Asset /scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifying.svg
        magnifying_glass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifying.svg")
        magnifying_glass.set_color(GLASS_COLOR)
        
        # Fixed: issue 27 - scale_factor=0.7
        self.place_at_grid(magnifying_glass, 'C5', scale_factor=0.7)
        # Offset slightly from the point initially
        magnifying_glass.move_to(target_point + np.array([0.8, 0.8, 0]))
        
        self.play(FadeIn(magnifying_glass))
        self.play(magnifying_glass.animate.move_to(target_point), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight Line 2
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(INTEGRAL_COLOR)
        )
        self.play(FadeOut(magnifying_glass), FadeOut(tangent_line), FadeOut(dot))
        
        # Paint Roller Asset
        # Fixed: issue 20 - use Asset /scratch/pawsey1357/jthen/Code2Video/assets/icon/roller.svg
        paint_roller = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/roller.svg")
        paint_roller.set_color(INTEGRAL_COLOR)
        paint_roller.scale(0.3) # Manual scaling for reasonable size
        
        start_x, end_x = 0.5, 3.5
        fill_tracker = ValueTracker(start_x)
        
        # Animated Area Fill
        # Persistent mobject with updater to avoid always_redraw overhead if possible, 
        # but for area get_area is the standard way.
        area = always_redraw(lambda: axes.get_area(curve, x_range=[start_x, fill_tracker.get_value()], color=INTEGRAL_COLOR, opacity=0.4))
        
        self.add(area)
        # Starting position for roller
        paint_roller.move_to(axes.c2p(start_x, curve_func(start_x)) + np.array([0, 0.4, 0]))
        
        self.play(FadeIn(paint_roller))
        
        # Use updater for smooth movement of roller
        def roller_updater(mob):
            curr_x = fill_tracker.get_value()
            curr_y = curve_func(curr_x)
            mob.move_to(axes.c2p(curr_x, curr_y) + np.array([0, 0.4, 0]))
            
        paint_roller.add_updater(roller_updater)
        self.play(fill_tracker.animate.set_value(end_x), run_time=3, rate_func=linear)
        paint_roller.remove_updater(roller_updater)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Reset and Highlight Line 3 (Using yellow for visual distinction)
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Show both aspects simultaneously for comparison
        self.play(
            FadeIn(dot),
            FadeIn(tangent_line),
            paint_roller.animate.move_to(self.grid['F6']),
            run_time=1.5
        )
        self.play(FadeOut(paint_roller))
        self.wait(2)
