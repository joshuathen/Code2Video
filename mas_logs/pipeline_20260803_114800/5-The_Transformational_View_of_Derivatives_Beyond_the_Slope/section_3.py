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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Visualizing f'(x) with Infinitesimal Intervals", [
            "Consider a tiny interval dx in the input.",
            "The function maps dx to a segment df.",
            "The derivative is the ratio of their lengths.",
            "For x squared, segments grow as x increases.",
            "At x equals 3, the output segment is 6-fold."
        ])

        # Colors
        COLOR_DX = "#FF0000"  # Bright Red
        COLOR_DF = "#FF0000"  # Bright Red
        COLOR_FORMULA = "#FFFFFF" # White
        COLOR_GROWTH = "#FFA500" # Orange
        COLOR_LINE = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Consider a tiny interval dx in the input.
        self.play(self.lecture[0].animate.set_color(COLOR_DX))
        
        input_line = Line(self.grid["B1"], self.grid["B6"], color=COLOR_LINE)
        input_label = Text("Input Space (x)", font_size=18, color=WHITE)
        self.place_in_area(input_label, "A1", "A3", scale_factor=0.8)
        
        # Small segment dx at x=3 (B3)
        dx_val = 0.2
        dx_seg = Line(
            self.grid["B3"] - RIGHT * dx_val/2, 
            self.grid["B3"] + RIGHT * dx_val/2, 
            color=COLOR_DX, 
            stroke_width=10
        )
        dx_text = MathTex("dx", color=COLOR_DX)
        # Issue 34 fix: place at A3
        self.place_at_grid(dx_text, "A3", scale_factor=0.7)
        
        self.play(Create(input_line), FadeIn(input_label))
        # Simulated "Zoom in" by scaling up the segment creation
        self.play(Create(dx_seg, run_time=1), FadeIn(dx_text))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The function maps dx to a segment df.
        self.play(self.lecture[1].animate.set_color(COLOR_DF))
        
        output_line = Line(self.grid["D1"], self.grid["D6"], color=COLOR_LINE)
        output_label = Text("Output Space (f(x))", font_size=18, color=WHITE)
        self.place_in_area(output_label, "E1", "E3", scale_factor=0.8)
        
        # At x=3, df length = f'(3) * dx = 6 * 0.2 = 1.2
        df_val = 1.2
        df_seg = Line(
            self.grid["D3"] - RIGHT * df_val/2, 
            self.grid["D3"] + RIGHT * df_val/2, 
            color=COLOR_DF, 
            stroke_width=10
        )
        df_text = MathTex("df", color=COLOR_DF)
        # Issue 35 fix: place at E3
        self.place_at_grid(df_text, "E3", scale_factor=0.7)
        
        mapping_line_l = DashedLine(dx_seg.get_left(), df_seg.get_left(), color=GRAY, stroke_opacity=0.5)
        mapping_line_r = DashedLine(dx_seg.get_right(), df_seg.get_right(), color=GRAY, stroke_opacity=0.5)
        
        self.play(Create(output_line), FadeIn(output_label))
        self.play(
            Create(mapping_line_l), 
            Create(mapping_line_r),
            TransformFromCopy(dx_seg, df_seg),
            FadeIn(df_text)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The derivative is the ratio of their lengths.
        self.play(self.lecture[2].animate.set_color(COLOR_FORMULA))
        
        formula = MathTex(r"f'(x) = \frac{df}{dx}", color=COLOR_FORMULA)
        self.place_at_grid(formula, "A5", scale_factor=1.0)
        
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # For x squared, segments grow as x increases.
        self.play(self.lecture[3].animate.set_color(COLOR_GROWTH))
        
        fx_label = MathTex("f(x) = x^2", color=COLOR_GROWTH)
        self.place_at_grid(fx_label, "B5", scale_factor=0.8)
        self.play(FadeIn(fx_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # At x equals 3, the output segment is 6-fold.
        self.play(self.lecture[4].animate.set_color(COLOR_DX))
        
        # Issue 36 fix: place at C5
        specific_ratio = MathTex(r"f'(3) = \frac{1.2}{0.2} = 6", color=COLOR_DX)
        self.place_at_grid(specific_ratio, "C5", scale_factor=0.8)
        
        self.play(Write(specific_ratio))
        # Pulse dx and df segments
        self.play(
            dx_seg.animate.set_stroke_width(15),
            df_seg.animate.set_stroke_width(15),
            rate_func=there_and_back,
            run_time=1.5
        )
        self.wait(2)
