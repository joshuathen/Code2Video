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

class Section3IntegralScene(TeachingScene):
    def construct(self):
        # Data
        title_text = "The Integral: Accumulating the 'Whole'"
        lecture_lines = [
            "An integral calculates the total accumulation over time.",
            "We slice the area into many thin rectangles.",
            "Summing these rectangles gives the total area.",
            "Notation \\int f(x) dx represents this infinite sum.",
            "It is like a rain gauge collecting every drop."
        ]
        
        # Setup
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        CLOUD_COLOR = "#87CEEB"
        GAUGE_COLOR = "#C0C0C0"
        CURVE_COLOR = "#4169E1"
        RECT_COLOR = "#ADD8E6"
        WATER_COLOR = "#1E90FF"
        NOTATION_COLOR = "#FFFFFF"

        # Assets
        cloud_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/cloud.svg"
        gauge_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/gauge.svg"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Cloud (Asset)
        cloud = SVGMobject(cloud_asset, color=CLOUD_COLOR, fill_opacity=1)
        self.place_in_area(cloud, 'A1', 'A4', scale_factor=0.7) # Fixed positioning (Issue 27)
        
        # Rain Gauge (Asset)
        gauge = SVGMobject(gauge_asset, color=GAUGE_COLOR, fill_opacity=1)
        self.place_in_area(gauge, "D5", "F6", scale_factor=0.9)
        
        # Graph
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 5, 1],
            x_length=3,
            y_length=2.5,
            axis_config={"include_tip": False, "font_size": 12}
        )
        curve = axes.plot(lambda x: 0.5 * (x - 2)**2 + 1, x_range=[0, 4.5], color=CURVE_COLOR)
        graph = VGroup(axes, curve)
        self.place_in_area(graph, "B1", "D4", scale_factor=0.8)
        
        self.play(FadeIn(cloud), FadeIn(gauge), Create(graph))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Thick rectangles
        rects_thick = axes.get_riemann_rectangles(
            curve, x_range=[0.5, 4.0], dx=0.7, 
            fill_opacity=0.5, color=RECT_COLOR, stroke_width=1
        )
        self.play(Create(rects_thick))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Thin rectangles
        rects_thin = axes.get_riemann_rectangles(
            curve, x_range=[0.5, 4.0], dx=0.15, 
            fill_opacity=0.6, color=RECT_COLOR, stroke_width=0.2
        )
        self.play(Transform(rects_thick, rects_thin))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Notation
        notation = MathTex(r"\int f(x) \, dx", color=NOTATION_COLOR)
        self.place_in_area(notation, 'E1', 'E4', scale_factor=1.0) # Fixed positioning (Issue 28)
        
        self.play(Write(notation))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Water fill inside gauge
        # Determine gauge bottom and width for water rectangle
        gauge_bottom = gauge.get_bottom()
        gauge_width = gauge.width * 0.8
        
        water = Rectangle(
            height=0.01, width=gauge_width, 
            fill_color=WATER_COLOR, fill_opacity=0.9, 
            stroke_width=0
        ).move_to(gauge_bottom, aligned_edge=DOWN)
        
        area_fill = axes.get_area(curve, x_range=[0.5, 4.0], color=WATER_COLOR, opacity=0.4)
        
        # Integral label near gauge
        integral_val_label = MathTex(r"\text{Total} = \int f(x) \, dx", font_size=20, color=NOTATION_COLOR)
        self.place_at_grid(integral_val_label, 'C5', scale_factor=0.7) # Fixed positioning (Issue 29)

        self.play(
            FadeOut(rects_thick),
            FadeIn(area_fill),
            FadeIn(water),
            water.animate.stretch_to_fit_height(gauge.height * 0.8).move_to(gauge_bottom, aligned_edge=DOWN),
            Write(integral_val_label),
            run_time=3
        )
        self.wait(2)
