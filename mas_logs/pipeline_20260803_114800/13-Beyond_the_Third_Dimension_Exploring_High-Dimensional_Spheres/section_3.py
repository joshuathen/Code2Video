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
        self.setup_layout("The Vanishing Volume Paradox", [
            "Intuition suggests volume grows with more dimensions.",
            "Surprisingly, n-sphere volume peaks at dimension five.",
            "Beyond five dimensions, the volume begins to shrink.",
            "As dimensions increase, the volume rapidly approaches zero.",
            "This is known as the vanishing volume paradox."
        ])
        
        # Pre-calculated unit n-sphere volumes for n=1 to 20
        volumes = [
            2.0, 3.14159, 4.18879, 4.93480, 5.26379, 
            5.16771, 4.72477, 4.05871, 3.29851, 2.55016, 
            1.88410, 1.33526, 0.91063, 0.59926, 0.38144, 
            0.23533, 0.14098, 0.08215, 0.04662, 0.02580
        ]
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        axes = Axes(
            x_range=[0, 21, 5],
            y_range=[0, 6, 1],
            axis_config={"color": "#BBBBBB"},
            x_length=4.5,
            y_length=3,
            tips=False
        )
        x_label = Text("Dimensions (n)", font_size=16, color="#BBBBBB")
        y_label = Text("Volume (V)", font_size=16, color="#BBBBBB")
        
        chart_group = VGroup(axes, x_label, y_label)
        # Resolved Issue 34: Move to A2 to avoid overlap with lecture lines
        self.place_in_area(chart_group, "A2", "D6")
        x_label.next_to(axes.x_axis, DOWN, buff=0.2)
        y_label.next_to(axes.y_axis, LEFT, buff=0.2).rotate(90*DEGREES)
        
        bars = VGroup()
        for i in range(5):
            bar_height = volumes[i] * (axes.y_length / 6)
            bar = Rectangle(
                width=0.12,
                height=bar_height,
                fill_color=BLUE,
                fill_opacity=0.8,
                stroke_width=1
            )
            bar.move_to(axes.c2p(i+1, 0), aligned_edge=DOWN)
            bars.add(bar)
            
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.play(Create(bars), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(YELLOW))
        
        # Highlight the 5th bar (n=5)
        self.play(bars[4].animate.set_color("#FFFF00").set_fill(opacity=1))
        peak_label = Text("Peak: n=5", font_size=14, color="#FFFF00")
        peak_label.next_to(bars[4], UP, buff=0.1)
        self.play(Write(peak_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(YELLOW))
        
        shrink_bars = VGroup()
        for i in range(5, 10):
            bar_height = volumes[i] * (axes.y_length / 6)
            bar = Rectangle(
                width=0.12,
                height=bar_height,
                fill_color=BLUE_E,
                fill_opacity=0.8,
                stroke_width=1
            )
            bar.move_to(axes.c2p(i+1, 0), aligned_edge=DOWN)
            shrink_bars.add(bar)
            
        self.play(Create(shrink_bars), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[2].animate.set_color(WHITE), self.lecture[3].animate.set_color(YELLOW))
        
        zero_bars = VGroup()
        for i in range(10, 20):
            bar_height = volumes[i] * (axes.y_length / 6)
            bar = Rectangle(
                width=0.12,
                height=bar_height,
                fill_color=DARK_GREY,
                fill_opacity=0.8,
                stroke_width=1
            )
            bar.move_to(axes.c2p(i+1, 0), aligned_edge=DOWN)
            zero_bars.add(bar)
            
        self.play(Create(zero_bars), run_time=1)
        
        # Resolved Issue 25: Use provided SVG asset
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg]
        sphere_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg")
        sphere_asset.set_color("#00CCFF")
        
        # Resolved Issue 35: Place at E3
        self.place_at_grid(sphere_asset, "E3", scale_factor=0.8)
        
        inf_label = Text("n → ∞", font_size=20, color=WHITE)
        inf_label.next_to(sphere_asset, DOWN, buff=0.1)
        
        self.play(FadeIn(sphere_asset), Write(inf_label))
        self.play(
            sphere_asset.animate.scale(0.1),
            inf_label.animate.set_color(YELLOW).scale(1.2),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[3].animate.set_color(WHITE), self.lecture[4].animate.set_color(YELLOW))
        
        paradox_text = Text("Volume to 0", font_size=32, color="#FF4444", weight=BOLD)
        # Resolved Issue 36: Place at E5
        self.place_at_grid(paradox_text, "E5", scale_factor=0.9)
        
        self.play(Write(paradox_text))
        self.wait(2)
