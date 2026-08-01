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
        # Setup title and lecture lines
        title = "Defining the Local Scaling Factor"
        lines = [
            "Focus on a tiny interval, dx, around a point.",
            "The function maps dx to output interval df.",
            "The derivative is the local scaling factor.",
            "Here, df equals the derivative times dx.",
            "Scaling factor determines if intervals stretch or shrink."
        ]
        self.setup_layout(title, lines)
        
        # Colors from storyboard
        INPUT_COLOR = "#00FF00"
        OUTPUT_COLOR = "#00BFFF"
        DX_COLOR = "#FFFF00"
        DF_COLOR = "#FF0000"
        TEXT_COLOR = "#FFFFFF"

        # Setup Visual Elements
        # Input Line (Green)
        # Positioned in area B2-B6 to maintain buffer from lecture notes (L003)
        input_line = NumberLine(x_range=[0, 10], length=4, color=INPUT_COLOR, include_tip=True)
        self.place_in_area(input_line, "B2", "B6")
        
        # Labels shifted to Col 2 to avoid crowding lecture notes (L003)
        input_label = Text("Input Line", font_size=18, color=INPUT_COLOR)
        self.place_at_grid(input_label, "A2", scale_factor=0.8)

        # Output Line (Blue)
        # Positioned in area D2-D6
        output_line = NumberLine(x_range=[0, 10], length=4, color=OUTPUT_COLOR, include_tip=True)
        self.place_in_area(output_line, "D2", "D6")
        
        output_label = Text("Output Line", font_size=18, color=OUTPUT_COLOR)
        self.place_at_grid(output_label, "C2", scale_factor=0.8)

        # ValueTracker for the derivative (scaling factor)
        deriv_tracker = ValueTracker(1.0)
        dx_val = 0.8 # Logical width on the number line

        # dx segment (Yellow)
        dx_seg = Line(
            input_line.n2p(5.0 - dx_val/2), 
            input_line.n2p(5.0 + dx_val/2), 
            color=DX_COLOR, stroke_width=6
        )
        dx_tag = Text("dx", font_size=20, color=DX_COLOR)
        dx_tag.next_to(dx_seg, UP, buff=0.1)

        # df segment (Red) - dynamic based on derivative value
        df_seg = always_redraw(lambda: Line(
            output_line.n2p(5.0 - (dx_val * deriv_tracker.get_value())/2),
            output_line.n2p(5.0 + (dx_val * deriv_tracker.get_value())/2),
            color=DF_COLOR, stroke_width=6
        ))
        
        df_tag = Text("df", font_size=20, color=DF_COLOR)
        df_tag.add_updater(lambda m: m.next_to(df_seg, DOWN, buff=0.1))

        # Initial add of background axes
        self.add(input_line, output_line, input_label, output_label)

        # === Animation for Lecture Line 1 ===
        # Line 1: Focus on a tiny interval, dx, around a point.
        self.play(self.lecture[0].animate.set_color(DX_COLOR))
        self.play(Create(dx_seg), Write(dx_tag))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line 2: The function maps dx to output interval df.
        self.play(self.lecture[1].animate.set_color(DF_COLOR))
        
        # Connectors (Dashed lines to show mapping)
        conn_l = always_redraw(lambda: DashedLine(
            dx_seg.get_left(), df_seg.get_left(), color=GRAY, stroke_opacity=0.4
        ))
        conn_r = always_redraw(lambda: DashedLine(
            dx_seg.get_right(), df_seg.get_right(), color=GRAY, stroke_opacity=0.4
        ))
        
        self.play(FadeIn(df_seg), Write(df_tag))
        self.play(Create(conn_l), Create(conn_r))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line 3: The derivative is the local scaling factor.
        self.play(self.lecture[2].animate.set_color(TEXT_COLOR))
        
        # Displaying the current scaling factor f'(x)
        deriv_label = Text("Scaling factor f'(x) =", font_size=22, color=TEXT_COLOR)
        deriv_val = DecimalNumber(deriv_tracker.get_value(), num_decimal_places=1, color=TEXT_COLOR)
        deriv_val.add_updater(lambda m: m.set_value(deriv_tracker.get_value()))
        deriv_disp = VGroup(deriv_label, deriv_val).arrange(RIGHT, buff=0.2)
        
        # FIX ISSUE 33: Optimize grid layout and centering
        self.place_in_area(deriv_disp, 'C2', 'C6', scale_factor=0.8)
        
        self.play(Write(deriv_disp))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Line 4: Here, df equals the derivative times dx.
        self.play(self.lecture[3].animate.set_color(TEXT_COLOR))
        
        # Core formula display
        formula = MathTex("df \\approx f'(x) \\cdot dx", color=TEXT_COLOR)
        
        # FIX ISSUE 34: Improve vertical separation and scaling
        self.place_in_area(formula, 'F1', 'F5', scale_factor=0.8)
        
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Line 5: Scaling factor determines if intervals stretch or shrink.
        self.play(self.lecture[4].animate.set_color(TEXT_COLOR))
        
        # Step 4: Stretch (f'(x) = 3)
        self.play(deriv_tracker.animate.set_value(3.0), run_time=2)
        self.wait(1)
        
        # Step 5: Shrink (f'(x) = 0.5)
        self.play(deriv_tracker.animate.set_value(0.5), run_time=2)
        self.wait(2)
