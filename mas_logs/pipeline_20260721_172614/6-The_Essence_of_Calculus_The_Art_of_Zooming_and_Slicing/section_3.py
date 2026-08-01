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
        # Initialize Layout
        self.setup_layout("Differentiation: The Power of 'Zooming In'", [
            "Differentiation helps us find speed at a single moment.",
            "Imagine zooming into a tiny point on a curve.",
            "As we zoom, the curve looks like a straight line.",
            "We can now calculate the slope of this flat line.",
            "This slope is the exact rate of change right there."
        ])

        # Initial state of lecture lines (all Gray to show progress)
        self.lecture.set_color(GRAY)

        # Colors from storyboard
        COLOR_CURVE = "#87CEEB"
        COLOR_HIGHLIGHT = "#FFFFFF"
        COLOR_TANGENT = "#FF0000"
        
        # Asset path
        ASSET_MAGNIFY = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/magn.svg"

        # === Animation for Lecture Line 1 ===
        # Show a smooth, complex curve in #87CEEB. Change line 1 color to #FFFFFF.
        self.play(self.lecture[0].animate.set_color(COLOR_HIGHLIGHT))
        
        axes_global = Axes(
            x_range=[-0.5, 3.5], y_range=[-0.5, 3.5],
            axis_config={"include_tip": False, "stroke_width": 2},
            x_length=4.5, y_length=3.5
        )
        
        def func(x):
            # Smooth curve
            return 0.5 * (x - 1.5)**2 + 0.5
        
        curve_global = axes_global.plot(func, x_range=[0.2, 3.3], color=COLOR_CURVE)
        global_group = VGroup(axes_global, curve_global)
        
        # Fix Issue 31: Move global_group to A2-C6 area to avoid obstruction
        self.place_in_area(global_group, "A2", "C6", scale_factor=0.8)
        
        self.play(Create(axes_global), Create(curve_global))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight a tiny point on the curve with a circle (#FFFFFF). Change line 2 color to #FFFFFF.
        self.play(self.lecture[1].animate.set_color(COLOR_HIGHLIGHT))
        
        p_x = 2.5
        p_y = func(p_x)
        point_p = Dot(axes_global.c2p(p_x, p_y), color=COLOR_HIGHLIGHT)
        highlight_circle = Circle(radius=0.15, color=COLOR_HIGHLIGHT).move_to(point_p)
        
        self.play(FadeIn(point_p), Create(highlight_circle))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Zoom into the point using the magnifying glass icon until the curve appears straight (#FFFFFF). 
        # Change line 3 color to #FFFFFF.
        self.play(self.lecture[2].animate.set_color(COLOR_HIGHLIGHT))
        
        # Load asset (Issue 24)
        magn_icon = SVGMobject(ASSET_MAGNIFY).set_color(COLOR_HIGHLIGHT)
        self.place_at_grid(magn_icon, "B5", scale_factor=0.5)
        
        # Zoomed view setup
        range_half = 0.05
        axes_zoom = Axes(
            x_range=[p_x - range_half, p_x + range_half],
            y_range=[p_y - range_half, p_y + range_half],
            axis_config={"include_tip": False, "stroke_width": 2},
            x_length=4, y_length=2.5
        )
        
        curve_zoom_segment = axes_zoom.plot(func, x_range=[p_x - range_half, p_x + range_half], color=COLOR_CURVE)
        straight_line = Line(
            axes_zoom.c2p(p_x - range_half, p_y - range_half),
            axes_zoom.c2p(p_x + range_half, p_y + range_half),
            color=COLOR_HIGHLIGHT
        )
        
        zoomed_area_group = VGroup(axes_zoom, curve_zoom_segment)
        self.place_in_area(zoomed_area_group, "D1", "F6")

        self.play(FadeIn(magn_icon))
        self.play(magn_icon.animate.move_to(point_p).scale(0.5))
        
        # Connectors for visual zoom effect
        conn_l = Line(highlight_circle.get_left(), self.grid["D1"], color=COLOR_HIGHLIGHT, stroke_width=1, stroke_opacity=0.5)
        conn_r = Line(highlight_circle.get_right(), self.grid["D6"], color=COLOR_HIGHLIGHT, stroke_width=1, stroke_opacity=0.5)

        self.play(
            Create(conn_l), Create(conn_r),
            FadeIn(axes_zoom), 
            FadeIn(curve_zoom_segment),
            magn_icon.animate.move_to(self.grid["D1"]).scale(0.8)
        )
        self.wait(1)
        self.play(ReplacementTransform(curve_zoom_segment, straight_line))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Draw a tangent line segment (#FF0000) and label 'Slope' (#FF0000). Change line 4 color to #FFFFFF.
        self.play(self.lecture[3].animate.set_color(COLOR_HIGHLIGHT))
        
        tangent_segment = Line(
            axes_zoom.c2p(p_x - 0.03, p_y - 0.03),
            axes_zoom.c2p(p_x + 0.03, p_y + 0.03),
            color=COLOR_TANGENT,
            stroke_width=6
        )
        
        slope_label = Text("Slope", color=COLOR_TANGENT, font_size=24)
        self.place_at_grid(slope_label, "E5", scale_factor=0.8)
        
        dx_val = 0.03
        p_start = axes_zoom.c2p(p_x, p_y)
        p_run = axes_zoom.c2p(p_x + dx_val, p_y)
        p_rise = axes_zoom.c2p(p_x + dx_val, p_y + dx_val)
        
        run_line = Line(p_start, p_run, color=COLOR_TANGENT)
        rise_line = Line(p_run, p_rise, color=COLOR_TANGENT)
        
        run_label = MathTex("dx", color=COLOR_TANGENT, font_size=20)
        rise_label = MathTex("dy", color=COLOR_TANGENT, font_size=20)
        
        # Fix Issue 33: Reposition run_label to E4
        self.place_at_grid(run_label, "E4", scale_factor=0.8)
        self.place_at_grid(rise_label, "D5", scale_factor=0.8)

        self.play(Create(tangent_segment), Write(slope_label))
        self.play(Create(run_line), Write(run_label), Create(rise_line), Write(rise_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Flash the tangent line to emphasize 'Rate of Change'. Change line 5 color to #FFFFFF.
        self.play(self.lecture[4].animate.set_color(COLOR_HIGHLIGHT))
        
        rate_of_change = MathTex(r"\frac{dy}{dx} = \text{Rate of Change}", color=COLOR_HIGHLIGHT, font_size=24)
        # Fix Issue 32: Place in area F4-F6 and scale to 0.7
        self.place_in_area(rate_of_change, "F4", "F6", scale_factor=0.7)
        
        self.play(Flash(tangent_segment, color=COLOR_HIGHLIGHT, line_length=0.3, num_lines=12))
        self.play(Write(rate_of_change))
        self.wait(3)
