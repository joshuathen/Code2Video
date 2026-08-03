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
        title_str = "Summary: The Language of the Universe"
        lecture_lines = [
            "PDEs describe everything from weather to quantum mechanics.",
            "They balance rates of change across our complex world.",
            "Master PDEs to decode the fundamental laws of nature."
        ]
        self.setup_layout(title_str, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Cycle through symbols: water drop, speaker, flame, spark.
        # Issue 32 fix: Use place_in_area(sym, 'B3', 'D5') to increase size/focus.
        self.play(self.lecture[0].animate.set_color(BLUE))
        
        # Define symbols as simple VGroups/Mobjects
        # Water drop
        water_drop = VGroup(
            Circle(fill_opacity=1, color="#1E90FF", radius=0.4),
            Triangle(fill_opacity=1, color="#1E90FF").scale(0.4).rotate(180*DEGREES).shift(UP*0.3)
        )
        # Speaker
        speaker = VGroup(
            Rectangle(height=0.6, width=0.3, fill_opacity=1, color="#808080"),
            Polygon([-0.1, 0.3, 0], [0.3, 0.5, 0], [0.3, -0.5, 0], [-0.1, -0.3, 0], fill_opacity=1, color="#808080")
        ).shift(LEFT*0.1)
        # Flame
        flame = Polygon([0, 0.6, 0], [-0.3, -0.2, 0], [0, 0, 0], [0.3, -0.2, 0], fill_opacity=1, color="#FF4500")
        # Spark
        spark = Star(n=8, inner_radius=0.15, outer_radius=0.5, fill_opacity=1, color="#FFFF00")

        symbols = [water_drop, speaker, flame, spark]
        for sym in symbols:
            # Applying fix for Issue 32
            self.place_in_area(sym, 'B3', 'D5', scale_factor=1.2)

        current_symbol = symbols[0]
        self.play(FadeIn(current_symbol))
        self.wait(0.5)

        for i in range(1, len(symbols)):
            self.play(ReplacementTransform(current_symbol, symbols[i]), run_time=0.8)
            current_symbol = symbols[i]
            self.wait(0.4)

        self.play(FadeOut(current_symbol))
        self.play(self.lecture[0].animate.set_color(WHITE))

        # === Animation for Lecture Line 2 ===
        # Display "Balancing Rates" with rotating gears #C0C0C0.
        # Issue 33 fix: Use place_in_area(balancing_text, 'C3', 'C5') for better positioning.
        self.play(self.lecture[1].animate.set_color(YELLOW))

        balancing_text = Text("Balancing Rates", font_size=28, color="#C0C0C0")
        # Applying fix for Issue 33
        self.place_in_area(balancing_text, 'C3', 'C5')
        
        # Helper to create a gear
        def create_gear(radius=0.5, color="#C0C0C0"):
            base = Circle(radius=radius, color=color, stroke_width=2)
            teeth = VGroup(*[
                Rectangle(width=radius*0.4, height=radius*0.2, color=color, fill_opacity=1, stroke_width=0)
                .move_to(base.point_at_angle(a * DEGREES))
                .rotate(a * DEGREES)
                for a in range(0, 360, 30)
            ])
            return VGroup(base, teeth)

        gear1 = create_gear(radius=0.55)
        gear2 = create_gear(radius=0.4)
        
        self.place_at_grid(gear1, "D2")
        self.place_at_grid(gear2, "E3")
        
        # Rotation logic using persistent mobjects and updaters
        gear1.add_updater(lambda m, dt: m.rotate(dt * 0.8))
        gear2.add_updater(lambda m, dt: m.rotate(-dt * 0.8 * (0.55/0.4))) # Synchronize based on tooth ratio
        
        self.play(Write(balancing_text), FadeIn(gear1), FadeIn(gear2))
        self.wait(2)
        
        self.play(FadeOut(balancing_text), FadeOut(gear1), FadeOut(gear2))
        self.play(self.lecture[1].animate.set_color(WHITE))

        # === Animation for Lecture Line 3 ===
        # Zoom out to show a dark blue starfield #191970.
        self.play(self.lecture[2].animate.set_color(PURPLE_A))
        
        starfield_bg = Rectangle(width=6, height=5, fill_color="#191970", fill_opacity=0.6, stroke_width=0)
        self.place_in_area(starfield_bg, "A1", "F6")
        
        # Get area bounds for random star distribution
        tl = self.grid["A1"]
        br = self.grid["F6"]
        
        # Create stars
        stars = VGroup(*[
            Dot(point=np.array([
                np.random.uniform(tl[0], br[0]),
                np.random.uniform(br[1], tl[1]),
                0
            ]), radius=0.03, color=WHITE)
            for _ in range(50)
        ])
        
        self.play(FadeIn(starfield_bg), Create(stars))
        
        # Zoom out effect: scale stars towards the center of the grid area
        center_point = (tl + br) / 2
        self.play(
            stars.animate.scale(0.4, about_point=center_point),
            starfield_bg.animate.scale(1.1),
            run_time=2.5
        )
        
        self.wait(1.5)
        self.play(FadeOut(stars), FadeOut(starfield_bg))
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
