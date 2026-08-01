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
        self.setup_layout(
            "Defining Euler's Number (e)", 
            [
                "Meet Euler's number: approximately 2.718.",
                "We call this special mathematical base e.",
                "At every point, slope exactly equals the height.",
                "The derivative of e to the x is itself.",
                "This synchronization makes e unique in calculus."
            ]
        )

        # === Animation for Lecture Line 1 ===
        # Draw a white #FFFFFF circular dial [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/dial.svg] 
        # with a needle and a label 'Base: 2.0'.
        self.lecture[0].set_color(WHITE)
        
        dial_asset = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/dial.svg").set_color(WHITE)
        needle = Line(ORIGIN, UP * 0.4, color=WHITE, stroke_width=4)
        # Position needle relative to dial center
        needle.move_to(dial_asset.get_center(), aligned_edge=DOWN)
        
        base_val = ValueTracker(2.0)
        dial_label = Text("Base: 2.0", font_size=24, color=WHITE)
        dial_label.next_to(dial_asset, DOWN, buff=0.1)
        
        dial_group = VGroup(dial_asset, needle, dial_label)
        self.place_in_area(dial_group, "A1", "B2", scale_factor=0.8)
        
        # Add updaters for the dial components
        def update_needle(mob):
            # Map base 2.0-3.0 to angle -45 to +45 degrees (roughly)
            angle = -(base_val.get_value() - 2.5) * PI / 2
            mob.set_angle(PI/2 + angle)
            mob.move_to(dial_asset.get_center(), aligned_edge=DOWN)

        def update_label(mob):
            val = base_val.get_value()
            if val < 2.71:
                mob.become(Text(f"Base: {val:.1f}", font_size=24, color=WHITE).next_to(dial_asset, DOWN, buff=0.1))
            elif 2.71 <= val <= 2.72:
                mob.become(Text("e ≈ 2.718", font_size=24, color="#FFD700").next_to(dial_asset, DOWN, buff=0.1))
            else:
                mob.become(Text(f"Base: {val:.1f}", font_size=24, color=WHITE).next_to(dial_asset, DOWN, buff=0.1))

        needle.add_updater(update_needle)
        dial_label.add_updater(update_label)

        self.play(Create(dial_asset), Create(needle), Write(dial_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Rotate the needle and update the label from 2.0 to 3.0, 
        # while a graph y = b^x morphs between 2^x and 3^x.
        self.lecture[1].set_color("#FFD700")

        axes = Axes(
            x_range=[-2, 2.5, 1],
            y_range=[-1, 8, 2],
            x_length=4,
            y_length=4,
            axis_config={"include_tip": True, "color": GREY_C},
        )
        labels = axes.get_axis_labels(
            x_label=Text("x", font_size=18), 
            y_label=Text("y", font_size=18)
        )
        axes_group = VGroup(axes, labels)
        self.place_in_area(axes_group, "A3", "D6", scale_factor=0.9)

        # Plot dynamic graph
        graph = always_redraw(lambda: axes.plot(
            lambda x: base_val.get_value()**x, 
            x_range=[-2, 2], 
            color=WHITE if base_val.get_value() < 2.71 or base_val.get_value() > 2.72 else "#FFD700"
        ))

        self.play(Create(axes), Write(labels))
        self.play(Create(graph))
        
        # Morphing base 2.0 -> 3.0
        self.play(base_val.animate.set_value(3.0), run_time=3)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Pause the needle at 2.718, changing the label to 'e ≈ 2.718' in gold #FFD700.
        # Draw a yellow #FFFF00 tangent line at x=0 with length 1 and a vertical height line of length 1.
        self.lecture[2].set_color("#FFFF00")

        self.play(base_val.animate.set_value(2.718), run_time=2)
        self.wait(1)

        # Tangent and height at x=0 for e^x
        # Height: from (0,0) to (0,1)
        # Slope: 1, so Tangent: y = x+1, draw from (-0.5, 0.5) to (0.5, 1.5)
        point_x = 0
        point_y = np.exp(point_x)
        
        height_line = Line(
            axes.c2p(0, 0), axes.c2p(0, point_y), 
            color="#FFFF00", stroke_width=6
        )
        tangent_line = Line(
            axes.c2p(-0.5, 0.5), axes.c2p(0.5, 1.5), 
            color="#FFFF00", stroke_width=6
        )
        
        height_label = Text("Height: 1.0", font_size=18, color="#FFFF00")
        slope_label = Text("Slope: 1.0", font_size=18, color="#FFFF00")
        metric_group = VGroup(height_label, slope_label).arrange(RIGHT, buff=0.5)
        self.place_in_area(metric_group, "F4", "F6", scale_factor=0.7)

        self.play(Create(height_line), Create(tangent_line))
        self.play(Write(metric_group))
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        # Display the formula 'd/dx(e^x) = e^x' in a gold #FFD700 box.
        self.lecture[3].set_color("#FFD700")

        formula_text = Text("d/dx(e^x) = e^x", font_size=32, color="#FFD700")
        formula_box = SurroundingRectangle(formula_text, color="#FFD700", buff=0.3)
        derivative_formula = VGroup(formula_text, formula_box)
        
        self.place_in_area(derivative_formula, "E1", "F3", scale_factor=0.8)
        
        self.play(Write(formula_text), Create(formula_box))
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        # This synchronization makes e unique in calculus.
        self.lecture[4].set_color(WHITE)
        
        # Highlight synchronization
        self.play(
            Indicate(height_line, scale_factor=1.2, color="#FFD700"),
            Indicate(tangent_line, scale_factor=1.2, color="#FFD700"),
            run_time=2
        )
        self.play(Indicate(derivative_formula, scale_factor=1.1, color=WHITE))
        self.wait(3)

        # Cleanup updaters
        needle.remove_updater(update_needle)
        dial_label.remove_updater(update_label)
