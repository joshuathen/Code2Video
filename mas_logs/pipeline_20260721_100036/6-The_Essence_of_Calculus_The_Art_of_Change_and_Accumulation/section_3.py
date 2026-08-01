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

class Section3Scene(TeachingScene):
    def construct(self):
        # Data from shared state
        title = "Differentiation: Zooming In for Truth"
        lecture_lines = [
            "Zooming into any smooth curve eventually reveals a line.",
            "This straight line represents the instantaneous rate of change.",
            "On a roller coaster, your direction is a tangent.",
            "We find this by narrowing the gap between points.",
            "This process of finding slopes is called differentiation."
        ]
        self.setup_layout(title, lecture_lines)

        # Assets
        rollercoaster_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/rollercoaster.svg"
        car_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/car.png"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFF00")
        
        # Rollercoaster curve asset [Asset: rollercoaster.svg]
        curve = SVGMobject(rollercoaster_asset).set_color("#00FFFF")
        # Issue 31 Fix: scale_factor=0.9
        self.place_in_area(curve, "A1", "F6", scale_factor=0.9)
        
        self.play(DrawBorderThenFill(curve))
        self.wait(0.5)
        
        # Zoom effect: Transition to a nearly straight line
        # We'll use a segment of a line for the "zoomed in" view
        line_start = self.grid["D2"]
        line_end = self.grid["C5"]
        zoomed_line = Line(start=line_start, end=line_end, color="#FFFF00")
        
        self.play(ReplacementTransform(curve, zoomed_line))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFF00")
        
        tangent_label = Text("Tangent", font_size=24, color="#FF8C00")
        # Issue 32 Fix: position B4, scale 0.8
        self.place_at_grid(tangent_label, "B4", scale_factor=0.8)
        
        self.play(Write(tangent_label))
        self.play(Flash(tangent_label, color="#FF8C00"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00")
        
        # Car icon asset [Asset: car.png]
        car = ImageMobject(car_asset).scale(0.2)
        car.move_to(line_start)
        # Orientation for the car
        angle = np.arctan2(line_end[1] - line_start[1], line_end[0] - line_start[0])
        car.rotate(angle)
        
        self.play(FadeIn(car))
        self.play(car.animate.move_to(line_end), run_time=2, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFFF00")
        
        # Transition back to a visible curve to show points converging
        # Using a simple curve to ensure point_from_proportion works reliably
        curve_context = CubicBezier(
            self.grid["E1"], self.grid["B2"], self.grid["E4"], self.grid["B6"]
        ).set_color("#00FFFF")
        # Issue 31 Fix: scale_factor=0.9
        # (Note: CubicBezier is already positioned by grid points, so scale_factor might shift it 
        # unless we scale it relative to its center after placement. 
        # But place_in_area moves it. Let's just create it and place it.)
        self.place_in_area(curve_context, "A1", "F6", scale_factor=0.9)
        
        self.play(
            FadeOut(zoomed_line, car, tangent_label),
            FadeIn(curve_context)
        )
        
        # Use ValueTracker for the moving point
        p1_prop = 0.5 
        p2_prop_tracker = ValueTracker(0.8)
        
        dot1 = Dot(curve_context.point_from_proportion(p1_prop), color=WHITE)
        # We'll use an updater for the second dot
        dot2 = Dot(color=WHITE)
        dot2.add_updater(lambda d: d.move_to(curve_context.point_from_proportion(p2_prop_tracker.get_value())))
        
        # Secant line between the two dots
        # Using a line that updates its start and end
        secant = always_redraw(lambda: Line(
            dot1.get_center(), 
            dot2.get_center(), 
            color="#FFFF00", 
            stroke_width=2
        ) if np.linalg.norm(dot1.get_center() - dot2.get_center()) > 0.05 else Line(
            dot1.get_center() + LEFT*0.5, 
            dot1.get_center() + RIGHT*0.5, 
            color="#FFFF00",
            stroke_width=2
        ).rotate(angle, about_point=dot1.get_center()))
        
        self.add(dot2) # Trigger updater
        self.play(Create(dot1), Create(secant))
        
        # Move dot2 to dot1
        self.play(p2_prop_tracker.animate.set_value(p1_prop), run_time=3)
        self.wait(1)
        
        dot2.clear_updaters()
        self.play(FadeOut(secant), FadeOut(dot1), FadeOut(dot2))
        self.wait(0.5)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFFF00")
        
        diff_text = Text("Differentiation", font_size=32, color="#FF8C00")
        # Issue 33 Fix: position F4, scale 0.9
        self.place_at_grid(diff_text, "F4", scale_factor=0.9)
        
        self.play(Write(diff_text))
        self.play(Flash(diff_text, color="#FF8C00"))
        self.wait(2)
