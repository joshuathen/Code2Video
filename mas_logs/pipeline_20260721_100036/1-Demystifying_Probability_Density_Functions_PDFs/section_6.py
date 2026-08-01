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

class Section6Scene(TeachingScene):
    def construct(self):
        # Setup the scene with specific title and lecture lines from the storyboard
        self.setup_layout("Summary & Key Takeaway", [
            "- PDFs map the likelihood landscape of continuous variables.",
            "- Remember the golden rule: Area always equals probability.",
            "- This tool unlocks understanding of our continuous world."
        ])
        
        # Define Colors
        COLOR_CURVE = "#D3D3D3"
        COLOR_GLOW = "#FFFF00"
        COLOR_TEXT = "#FFFFFF"

        # Gaussian function for visualization
        def pdf_func(x):
            return 1.5 * np.exp(-0.5 * (x / 0.8)**2)

        # === Animation for Lecture Line 1 ===
        # Line: "PDFs map the likelihood landscape of continuous variables."
        # Action: Display various glowing PDF shapes.
        self.play(self.lecture[0].animate.set_color(COLOR_CURVE))
        
        # Create Axes for the landscape
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[0, 2, 0.5],
            x_length=5,
            y_length=4,
            axis_config={"include_tip": False, "color": GRAY_E}
        )
        # Positioning axes in the B1-F6 area
        self.place_in_area(axes, "B1", "F6")
        
        # Main Curve
        curve = axes.plot(pdf_func, color=COLOR_CURVE, stroke_width=4)
        
        # Background curves (slightly offset and faded) to create a 'landscape' feel
        def pdf_func_2(x): return 1.0 * np.exp(-0.5 * ((x-1.5) / 0.5)**2)
        def pdf_func_3(x): return 0.7 * np.exp(-0.5 * ((x+1.8) / 0.7)**2)
        
        curve_bg1 = axes.plot(pdf_func_2, color=COLOR_CURVE).set_stroke(opacity=0.3)
        curve_bg2 = axes.plot(pdf_func_3, color=COLOR_CURVE).set_stroke(opacity=0.2)

        self.play(Create(axes), run_time=1)
        self.play(Create(curve), Create(curve_bg1), Create(curve_bg2), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line: "Remember the golden rule: Area always equals probability."
        # Action: Highlight a slice; "Area = Probability" glows.
        self.play(self.lecture[1].animate.set_color(COLOR_GLOW))
        
        # Define the slice area
        area_slice = axes.get_area(curve, x_range=[0.3, 1.3], color=COLOR_GLOW, opacity=0.4)
        
        # Text "Area = Probability" - positioned at B5 to avoid overlap
        area_label = Text("Area = Probability", font_size=24, color=COLOR_GLOW)
        self.place_at_grid(area_label, "B5", scale_factor=0.8)
        
        # Highlight the curve segment for the slice
        curve_highlight = axes.plot(pdf_func, x_range=[0.3, 1.3], color=COLOR_GLOW, stroke_width=6)
        
        self.play(
            FadeIn(area_slice),
            Create(curve_highlight),
            Write(area_label),
            run_time=1.5
        )
        self.play(Indicate(area_label, color=COLOR_GLOW, scale_factor=1.1))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line: "This tool unlocks understanding of our continuous world."
        # Action: Text "Mapping the Likelihood Landscape" fades in alongside the map icon [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/map.svg].
        self.play(self.lecture[2].animate.set_color(COLOR_TEXT))
        
        # Load asset
        map_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/map.svg", color=COLOR_TEXT)
        landscape_text = Text("Mapping the Likelihood Landscape", font_size=28, color=COLOR_TEXT)
        
        # Group icon and text
        final_group = VGroup(map_icon, landscape_text).arrange(RIGHT, buff=0.3)
        
        # Place in area A1 to A6 for centering and visibility
        self.place_in_area(final_group, "A1", "A6", scale_factor=0.8)
        
        self.play(FadeIn(final_group, shift=UP))
        self.wait(3)
