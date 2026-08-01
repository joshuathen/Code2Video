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

class Section4Scene(TeachingScene):
    def construct(self):
        title_text = "The Two Golden Rules"
        lecture_lines = [
            "A PDF must never fall below the horizontal axis.",
            "Negative probability is impossible in the real world.",
            "The total area under the curve must equal one.",
            "One represents a one-hundred percent chance of any outcome.",
            "The \"Magic Carpet\" always covers exactly one unit."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        COLOR_CURVE = "#FFB6C1"
        COLOR_NO_NEG = "#FF0000"
        COLOR_SHADE = "#E6E6FA"
        COLOR_ONE = "#FFFFFF"
        COLOR_GLOW = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        # Apply color change to lecture line
        self.play(self.lecture[0].animate.set_color(COLOR_CURVE))
        
        # Create Axes manually to fit the area
        # Fix for Issue 31: Moving axes from B1 to B2 to avoid crowding
        axes = Axes(
            x_range=[0, 6, 1],
            y_range=[0, 3, 1],
            x_length=5.0, # Slightly shortened length to fit B2-E6 better
            y_length=3.5,
            axis_config={"include_tip": True, "color": GRAY}
        )
        self.place_in_area(axes, "B2", "E6")
        
        # Define a bell curve function strictly above x-axis
        def curve_func(x):
            # A Gaussian shape that stays positive
            return 2.5 * np.exp(-0.5 * (x - 3)**2) + 0.1
            
        curve = axes.plot(curve_func, x_range=[0, 6], color=COLOR_CURVE)
        
        # Show curve strictly above the x-axis
        self.play(Create(axes), Create(curve), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Apply color change to lecture line
        self.play(self.lecture[1].animate.set_color(COLOR_NO_NEG))
        
        # "No Negative" symbol below x-axis
        # Fix for Issue 32: Moved to F4 and increased scale_factor to 1.0
        no_neg_text = MathTex(r"P(x) < 0", color=COLOR_NO_NEG)
        self.place_at_grid(no_neg_text, "F4", scale_factor=1.0)
        
        cross = Cross(no_neg_text, stroke_color=COLOR_NO_NEG, stroke_width=6)
        
        # Display "No Negative" symbol then cross it out
        self.play(Write(no_neg_text))
        self.play(Create(cross))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Apply color change to lecture line
        self.play(self.lecture[2].animate.set_color(COLOR_SHADE))
        
        # Shade entire area under the curve
        area = axes.get_area(curve, x_range=[0, 6], color=COLOR_SHADE, opacity=0.5)
        
        self.play(FadeIn(area))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Apply color change to lecture line
        self.play(self.lecture[3].animate.set_color(COLOR_ONE))
        
        # 100% label transforms into 1
        # Fix for Issue 33: Increasing scale_factor to 1.2
        label_100 = Text("100%", color=COLOR_ONE, font_size=32)
        self.place_at_grid(label_100, "B5", scale_factor=1.2)
        
        label_1 = Text("1", color=COLOR_ONE, font_size=48)
        self.place_at_grid(label_1, "B5", scale_factor=1.2)
        
        # Display "100%" label that transforms into a large "1"
        self.play(Write(label_100))
        self.wait(0.5)
        self.play(Transform(label_100, label_1))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Apply color change to lecture line
        self.play(self.lecture[4].animate.set_color(COLOR_GLOW))
        
        # Show the entire shaded area (the "Magic Carpet") glow intensely
        glow_area = axes.get_area(curve, x_range=[0, 6], color=COLOR_GLOW, opacity=0.3)
        
        self.play(
            area.animate.set_color(COLOR_GLOW).set_opacity(0.7),
            FadeIn(glow_area),
            run_time=1
        )
        self.play(
            Indicate(area, color=COLOR_GLOW, scale_factor=1.05),
            run_time=1.5
        )
        self.wait(2)
