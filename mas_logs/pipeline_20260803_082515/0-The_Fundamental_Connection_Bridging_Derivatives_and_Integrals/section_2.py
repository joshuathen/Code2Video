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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisite Refresh: Slope vs. Area", [
            "Derivatives measure the instantaneous slope of a curve.",
            "Integrals calculate the accumulated area under a curve.",
            "These concepts seem distinct but are deeply linked."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Highlight Line 1
        self.play(self.lecture[0].animate.set_color("#FF0000"))
        
        # Load Speedometer Asset
        speedometer_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/speedometer.svg").set_color(WHITE)
        # Assuming the center of the gauge is near the origin of the SVG
        needle = Arrow(ORIGIN, UP * 1.0, buff=0, color="#FF0000")
        needle_pivot = Dot(color="#FF0000", radius=0.08)
        
        speedometer = VGroup(speedometer_svg, needle, needle_pivot)
        # Fix for Issue 26: Scale factor to 0.8
        self.place_in_area(speedometer, "A1", "C3", scale_factor=0.8)
        
        # Align needle to the base of the speedometer (assuming centered pivot for common speedometer icons)
        needle.move_to(speedometer_svg.get_center())
        needle_pivot.move_to(speedometer_svg.get_center())
        needle.rotate(PI/2, about_point=needle.get_start()) # Start at left (0 speed)
        
        self.play(FadeIn(speedometer_svg), FadeIn(needle_pivot))
        self.play(Create(needle))
        self.play(Rotate(needle, angle=-PI/2, about_point=needle.get_start()), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight Line 2
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FFFF00")
        )
        
        # Create Axes and Curve
        axes = Axes(
            x_range=[0, 4, 1], y_range=[0, 4, 1], 
            axis_config={"include_tip": True},
            x_length=3, y_length=3
        ).set_color(WHITE)
        curve = axes.plot(lambda x: 0.2 * x**2 + 0.5, x_range=[0, 3.5], color="#FFFFFF")
        
        # Shaded area tracker
        area_tracker = ValueTracker(0.01)
        area = always_redraw(lambda: axes.get_area(curve, x_range=[0, area_tracker.get_value()], color="#FFFF00", opacity=0.5))
        
        graph_group = VGroup(axes, curve)
        # Fix for Issue 26: Scale factor to 0.8
        self.place_in_area(graph_group, "D1", "F3", scale_factor=0.8)
        
        # Add area after the axes are positioned
        self.add(area)
        
        self.play(Create(axes), Create(curve))
        self.play(area_tracker.animate.set_value(3.5), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight Line 3
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Labels for concepts
        label_slope = Text("Slope", color="#FF0000", font_size=24)
        label_area = Text("Area", color="#FFFF00", font_size=24)
        
        # Fix for Issue 27 and 28: Positioning and scaling
        self.place_at_grid(label_slope, "B4", scale_factor=1.2)
        self.place_at_grid(label_area, "E4", scale_factor=1.2)
        
        self.play(Write(label_slope), Write(label_area))
        
        # Highlight both visual elements to show connection
        self.play(
            Indicate(needle, color="#FF0000"),
            Indicate(area, color="#FFFF00"),
            run_time=2
        )
        self.wait(2)
