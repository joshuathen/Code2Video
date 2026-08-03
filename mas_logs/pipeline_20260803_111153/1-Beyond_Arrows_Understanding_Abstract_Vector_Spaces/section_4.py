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
        lecture_lines = [
            "- Consider functions like sine waves as abstract vectors.",
            "- Adding two functions creates a new, combined wave.",
            "- Multiplying a function by a scalar changes its amplitude.",
            "- Because they follow axioms, functions form a vector space.",
            "- This abstraction allows us to treat graphs like arrows."
        ]
        self.setup_layout("Case Study: Functions as Vectors", lecture_lines)

        # Assets
        SINE_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/sine.svg"

        # Colors
        COLOR_SINE = "#FF0000"    # Red
        COLOR_COSINE = "#0000FF"  # Blue
        COLOR_SUM = "#FFFF00"     # Yellow
        COLOR_ARROW = "#00FF00"   # Green
        COLOR_TEXT = WHITE

        # === Animation for Lecture Line 1 ===
        # Draw a sine wave (#FF0000) [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/sine.svg] 
        # and a cosine wave (#0000FF) on axes, labeled as vectors.
        self.lecture[0].set_color(COLOR_SINE)
        
        axes = Axes(
            x_range=[-PI, PI, PI/2],
            y_range=[-2, 2, 1],
            x_length=4.5,
            y_length=3.5,
            axis_config={"include_tip": True, "color": GREY_C}
        )
        self.place_in_area(axes, "B2", "E6")
        
        # Sine wave from asset - colored Red as per storyboard
        sine_wave = SVGMobject(SINE_ASSET).set_color(COLOR_SINE)
        self.place_in_area(sine_wave, "B2", "E6", scale_factor=0.6)
        
        # Cosine wave as plot for context
        cosine_graph = axes.plot(lambda x: np.cos(x), color=COLOR_COSINE)
        
        sine_label = MathTex("f(x)", color=COLOR_SINE, font_size=24)
        cosine_label = MathTex("g(x)", color=COLOR_COSINE, font_size=24)
        
        self.place_at_grid(sine_label, "B2", scale_factor=1.0)
        # Issue 29 Fix: Move cosine_label to B4 to avoid overlap
        self.place_at_grid(cosine_label, "B4", scale_factor=1.0)

        self.play(Create(axes))
        self.play(DrawBorderThenFill(sine_wave), Write(sine_label))
        self.play(Create(cosine_graph), Write(cosine_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Overlay the waves and draw the sum wave (#FFFF00) through point-wise addition.
        self.lecture[0].set_color(COLOR_TEXT)
        self.lecture[1].set_color(COLOR_SUM)
        
        sum_graph = axes.plot(lambda x: np.sin(x) + np.cos(x), color=COLOR_SUM)
        sum_label = MathTex("(f+g)(x)", color=COLOR_SUM, font_size=24)
        self.place_at_grid(sum_label, "B5", scale_factor=1.0)
        
        self.play(Create(sum_graph), Write(sum_label))
        self.play(FadeOut(cosine_graph), FadeOut(cosine_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Stretch the sine wave (#FF0000) vertically to demonstrate scaling by a scalar.
        self.lecture[1].set_color(COLOR_TEXT)
        self.lecture[2].set_color(COLOR_SINE)
        
        # Scaling effect: stretch the SVG
        scaled_sine_wave = sine_wave.copy().stretch(1.5, dim=1)
        scaled_label = MathTex("1.5 \\cdot f(x)", color=COLOR_SINE, font_size=24)
        self.place_at_grid(scaled_label, "C2", scale_factor=1.0)
        
        self.play(
            ReplacementTransform(sine_wave, scaled_sine_wave),
            ReplacementTransform(sine_label, scaled_label),
            FadeOut(sum_graph),
            FadeOut(sum_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Display an axiom checklist next to the waves with green checkmarks (#00FF00) appearing.
        self.lecture[2].set_color(COLOR_TEXT)
        self.lecture[3].set_color(COLOR_TEXT) 
        
        # Issue 28 Fix: Move axioms_box to F5, scale 0.8 to avoid obstructing curves
        axioms_box = VGroup(
            Text("8 Vector Axioms", font_size=24, color=WHITE),
            Text("✓ Closure", font_size=20, color=GREEN),
            Text("✓ Addition", font_size=20, color=GREEN),
            Text("✓ Scaling", font_size=20, color=GREEN)
        ).arrange(DOWN, aligned_edge=LEFT)
        
        self.place_at_grid(axioms_box, "F5", scale_factor=0.8)
        
        vs_name = Text("Space: C[a,b]", font_size=28, color=YELLOW)
        self.place_at_grid(vs_name, "F4", scale_factor=0.8)
        
        self.play(Write(axioms_box))
        self.play(Write(vs_name))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Morph a 2D arrow vector (#00FF00) into a sine wave graph (#FF0000) [Asset: ...]
        self.lecture[3].set_color(COLOR_TEXT)
        self.lecture[4].set_color(COLOR_ARROW)
        
        # Issue 30 Fix: arrow_vec scale 0.9 to avoid edge-cutting
        arrow_vec = Arrow(
            start=LEFT*0.8, end=RIGHT*0.8,
            color=COLOR_ARROW,
            stroke_width=8
        )
        self.place_in_area(arrow_vec, "B2", "E6", scale_factor=0.9)
        
        self.play(
            FadeOut(axioms_box),
            FadeOut(vs_name),
            FadeOut(scaled_label),
            FadeOut(axes),
            FadeOut(scaled_sine_wave)
        )
        
        # Morphing: arrow into the sine wave asset
        # Issue 22: Integration of SVG asset
        final_sine_svg = SVGMobject(SINE_ASSET).set_color(COLOR_SINE)
        self.place_in_area(final_sine_svg, "B2", "E6", scale_factor=0.6)
        
        self.play(Create(arrow_vec))
        self.wait(0.5)
        self.play(ReplacementTransform(arrow_vec, final_sine_svg))
        self.wait(2)
