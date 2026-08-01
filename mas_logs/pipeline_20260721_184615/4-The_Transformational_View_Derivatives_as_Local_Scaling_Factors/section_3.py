from manim import *
import numpy as np

# Use the provided TeachingScene base class without modification.
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
        # Fetch storyboard title and lines
        title = "The Local Zoom: Non-Linear Transformations"
        lines = [
            "Non-linear functions stretch space differently at every point.",
            "Zooming in reveals that the local behavior becomes linear.",
            "At x equals one, the interval doubles its length."
        ]
        self.setup_layout(title, lines)

        # Colors - using hex strings as per [L008]
        color_curve = "#FFFFFF"  # White
        color_seg1 = "#00FF00"   # Green for segment at x=0.5
        color_seg2 = "#FFA500"   # Orange for segment at x=1.5
        color_zoom = "#87CEEB"   # Sky Blue for segment at x=1.0
        color_output = "#FF6347" # Tomato for mapped output segment
        color_result = "#FFFF00" # Yellow for final derivative label

        # === Animation for Lecture Line 1 ===
        # Non-linear functions stretch space differently at every point.
        self.play(self.lecture[0].animate.set_color(color_curve))
        
        axes = Axes(
            x_range=[0, 2, 1],
            y_range=[0, 4, 1],
            x_length=3.5,
            y_length=3.5,
            axis_config={"include_tip": True, "color": "#FFFFFF"}
        )
        curve = axes.plot(lambda x: x**2, x_range=[0, 2], color=color_curve)
        func_label = MathTex("f(x) = x^2", font_size=28, color=color_curve)
        
        graph_group = VGroup(axes, curve)
        self.place_in_area(graph_group, "B2", "E5")
        self.place_at_grid(func_label, "B5", scale_factor=0.8) # [L002]
        
        # Segments at x=0.5 and x=1.5 to show varying steepness
        seg_05 = axes.plot(lambda x: x**2, x_range=[0.4, 0.6], color=color_seg1, stroke_width=8)
        seg_15 = axes.plot(lambda x: x**2, x_range=[1.4, 1.6], color=color_seg2, stroke_width=8)
        
        self.play(Create(axes), Create(curve), FadeIn(func_label))
        self.play(Create(seg_05), Create(seg_15))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # Zooming in reveals that the local behavior becomes linear.
        self.play(self.lecture[1].animate.set_color(color_zoom))
        
        # Highlight target segment at x=1.0
        seg_10 = axes.plot(lambda x: x**2, x_range=[0.9, 1.1], color=color_zoom, stroke_width=8)
        
        self.play(FadeOut(seg_05), FadeOut(seg_15))
        self.play(Create(seg_10))
        self.wait(0.5)

        # Local Zoom: Create zoomed-in axes
        zoomed_axes = Axes(
            x_range=[0.95, 1.05, 0.05],
            y_range=[0.9, 1.1, 0.1],
            x_length=4,
            y_length=4,
            axis_config={"include_tip": False, "color": "#FFFFFF"}
        )
        zoomed_curve = zoomed_axes.plot(lambda x: x**2, x_range=[0.95, 1.05], color=color_zoom)
        self.place_in_area(zoomed_axes, "B2", "E5")

        self.play(
            FadeOut(graph_group),
            FadeOut(seg_10),
            FadeOut(func_label),
            ReplacementTransform(seg_10.copy(), zoomed_curve),
            Create(zoomed_axes),
            run_time=2
        )
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # At x equals one, the interval doubles its length.
        self.play(self.lecture[2].animate.set_color(color_result))
        
        # Transition to Mapping View (Parallel Number Lines)
        self.play(FadeOut(zoomed_axes), FadeOut(zoomed_curve))

        # Setup Number Lines
        input_line = NumberLine(x_range=[0.98, 1.02, 0.01], length=5, include_numbers=True, font_size=20, color="#FFFFFF")
        output_line = NumberLine(x_range=[0.96, 1.04, 0.02], length=5, include_numbers=True, font_size=20, color="#FFFFFF")
        
        self.place_at_grid(input_line, "C4")
        self.place_at_grid(output_line, "E4")
        
        input_label = Text("Input x", font_size=24, color="#FFFFFF")
        output_label = Text("Output f(x)", font_size=24, color="#FFFFFF")
        
        # Addressing Issue 24: Move labels closer to lines (C3 instead of C2, E3 instead of E2)
        self.place_at_grid(input_label, "C3", scale_factor=0.8)
        self.place_at_grid(output_label, "E3", scale_factor=0.8)
        
        self.play(Create(input_line), Create(output_line), FadeIn(input_label), FadeIn(output_label))
        
        # Segment Mapping: [1, 1.01] -> [1, 1.0201]
        x1, x2 = 1.0, 1.01
        y1, y2 = 1.0, 1.0201 # x^2
        
        in_seg = Line(input_line.n2p(x1), input_line.n2p(x2), color=color_zoom, stroke_width=10)
        out_seg = Line(output_line.n2p(y1), output_line.n2p(y2), color=color_output, stroke_width=10)
        
        # Connection lines
        conn1 = DashedLine(input_line.n2p(x1), output_line.n2p(y1), color="#FFFFFF", stroke_opacity=0.5)
        conn2 = DashedLine(input_line.n2p(x2), output_line.n2p(y2), color="#FFFFFF", stroke_opacity=0.5)
        
        dx_label = MathTex(r"\Delta x = 0.01", font_size=22, color=color_zoom)
        df_label = MathTex(r"\Delta f \approx 0.02", font_size=22, color=color_output)
        
        self.place_at_grid(dx_label, "B4", scale_factor=0.8)
        self.place_at_grid(df_label, "F4", scale_factor=0.8)
        
        self.play(Create(in_seg), Create(out_seg), Create(conn1), Create(conn2))
        self.play(FadeIn(dx_label), FadeIn(df_label))
        self.wait(1)
        
        # Addressing Issue 25: Move ratio calculation to D5 to avoid overlap with connection lines
        ratio = MathTex(r"\text{Scale} \approx \frac{0.02}{0.01} = 2", font_size=28, color=color_result)
        self.place_at_grid(ratio, "D5")
        
        self.play(Write(ratio))
        self.wait(1)
        
        # Final Derivative result
        # Addressing Issue 26: Move formula to F6 to avoid crowding near df_label at F4
        deriv_formula = MathTex(r"f'(1) = 2", font_size=36, color=color_result)
        self.place_at_grid(deriv_formula, "F6")
        
        self.play(Write(deriv_formula))
        self.play(Indicate(deriv_formula, color=color_result)) # [L004]
        self.wait(3)
