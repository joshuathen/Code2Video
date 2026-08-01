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

class Section7Scene(TeachingScene):
    def construct(self):
        title_text = "Real-World Impact: From RAM to Deep Space"
        lecture_lines = [
            "Hamming codes protect modern server memory from crashes.",
            "They allow deep-space probes to communicate reliably.",
            "This math ensures your bank balance remains accurate."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Animate a green RAM stick [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/ram.svg] (#00FF00) with flowing binary data.
        # Change Line 1 color to #FFFF00. self.wait(2).
        
        # Load RAM SVG Asset - Resolve Issue 28
        ram_stick = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ram.svg")
        ram_stick.set_color("#00FF00")
        self.place_in_area(ram_stick, "B2", "C5", scale_factor=1.5)
        
        # Flowing binary data - Issue 44: Move to A3 to avoid overlap with B2-C5 area
        binary_data = VGroup(*[
            Text(str(np.random.choice(["0", "1"])), font_size=20, color="#FFFFFF")
            for _ in range(8)
        ]).arrange(RIGHT, buff=0.3)
        self.place_at_grid(binary_data, "A3", scale_factor=0.8) # L002: scale 0.8
        
        self.play(
            self.lecture[0].animate.set_color("#FFFF00"),
            FadeIn(ram_stick),
            FadeIn(binary_data)
        )
        
        # Binary data flow animation
        binary_data.add_updater(lambda m, dt: m.shift(RIGHT * 0.5 * dt))
        self.wait(2.0)
        
        binary_data.clear_updaters()
        self.play(FadeOut(ram_stick), FadeOut(binary_data))

        # === Animation for Lecture Line 2 ===
        # Show a satellite (#AAAAAA) beaming a signal to a distant Earth icon.
        # Change Line 2 color to #FFFF00. self.wait(2).
        
        # Satellite visual - Issue 45: place at C2
        sat_body = Square(side_length=0.6, fill_opacity=1, fill_color="#AAAAAA")
        sat_panel_l = Rectangle(height=0.4, width=0.8, fill_opacity=1, fill_color="#0000FF").next_to(sat_body, LEFT, buff=0)
        sat_panel_r = Rectangle(height=0.4, width=0.8, fill_opacity=1, fill_color="#0000FF").next_to(sat_body, RIGHT, buff=0)
        satellite = VGroup(sat_body, sat_panel_l, sat_panel_r)
        self.place_at_grid(satellite, "C2", scale_factor=0.8)
        
        # Earth visual
        earth = Circle(radius=0.5, fill_opacity=1, fill_color="#0000FF", stroke_color="#FFFFFF")
        earth_land = VGroup(
            Triangle().scale(0.2).set_fill("#00FF00", 1).move_to(earth.get_center() + UP*0.1),
            Square().scale(0.15).set_fill("#00FF00", 1).move_to(earth.get_center() + DOWN*0.1 + RIGHT*0.1)
        )
        earth_group = VGroup(earth, earth_land)
        self.place_at_grid(earth_group, "E5", scale_factor=0.9)
        
        # Signal visual
        signal_base = Arc(radius=0.3, start_angle=-TAU/8, angle=TAU/4, color="#FFFFFF")
        signal_base.rotate(-TAU/8) # Aim towards E5 roughly
        
        self.play(
            self.lecture[1].animate.set_color("#FFFF00"),
            FadeIn(satellite),
            FadeIn(earth_group)
        )
        
        # Beaming animation
        signals = VGroup()
        # Vector from C2 to E5 is approx (3, -2)
        for i in range(3):
            s = signal_base.copy()
            s.move_to(satellite.get_center() + RIGHT * (i+1) * 0.8 + DOWN * (i+1) * 0.5)
            signals.add(s)
            
        self.play(
            LaggedStart(*[FadeIn(s) for s in signals], lag_ratio=0.5),
            run_time=2
        )
        
        self.wait(2.0)
        self.play(FadeOut(satellite), FadeOut(earth_group), FadeOut(signals))

        # === Animation for Lecture Line 3 ===
        # Display a large white "1011" with a green checkmark (#00FF00) overlay.
        # Change Line 3 color to #FFFF00. self.wait(2).
        
        # Issue 46: Move balance_text to area E3-F5
        balance_text = Text("1011", font_size=60, color="#FFFFFF")
        self.place_in_area(balance_text, "E3", "F5", scale_factor=1.0)
        
        # Green checkmark
        checkmark = VGroup(
            Line(LEFT * 0.2 + DOWN * 0.2, ORIGIN, color="#00FF00", stroke_width=8),
            Line(ORIGIN, RIGHT * 0.4 + UP * 0.5, color="#00FF00", stroke_width=8)
        )
        checkmark.next_to(balance_text, RIGHT, buff=0.3)
        
        self.play(
            self.lecture[2].animate.set_color("#FFFF00"),
            Write(balance_text)
        )
        self.play(Create(checkmark))
        # Use Indicate as per L004
        self.play(Indicate(balance_text, color="#00FF00"))
        
        self.wait(2.0)
