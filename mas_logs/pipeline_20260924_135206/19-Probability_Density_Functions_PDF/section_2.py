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
        self.setup_layout("Visualizing the Curve", [
            "PDFs track probability density over continuous domains.",
            "Histograms with narrow bins approximate smooth curves.",
            "The total area under the curve equals one."
        ])
        
        # Axes
        axes = Axes(x_range=[0, 6, 1], y_range=[0, 1.5, 0.5], x_length=4, y_length=3)
        self.place_in_area(axes, "B2", "D5", scale_factor=0.5)
        
        # PDF curve
        pdf_curve = axes.plot(lambda x: (x**2) * np.exp(-x) * 1.5, x_range=[0, 6], color=WHITE)
        area_fill = axes.get_area(pdf_curve, [0, 6], color="#40E0D0", opacity=0.3)
        f_x_label = MathTex("f(x)", color="#00FFFF")
        self.place_at_grid(f_x_label, "D4", scale_factor=0.7)
        
        # Histogram bars (using Asset)
        hist_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/histogram.svg")
        hist_asset.set_color("#FFD700")
        self.place_in_area(hist_asset, "B2", "D5", scale_factor=0.3)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFD700")
        self.add(axes, hist_asset)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFD700")
        self.play(ReplacementTransform(hist_asset, pdf_curve))
        self.play(Write(f_x_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FFD700")
        self.play(FadeIn(area_fill))
        
        # Re-reference the asset for the final flash
        hist_icon_ref = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/histogram.svg")
        hist_icon_ref.set_color("#FF00FF")
        self.place_at_grid(hist_icon_ref, "C4", scale_factor=0.2)
        
        area_text = Text("Area = 1", color="#FF00FF")
        self.place_at_grid(area_text, "E4", scale_factor=0.9)
        
        self.play(Flash(area_text, color="#FF00FF"))
        self.wait(2)
