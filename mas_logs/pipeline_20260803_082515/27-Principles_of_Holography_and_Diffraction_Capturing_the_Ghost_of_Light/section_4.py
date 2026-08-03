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
        # Initial Setup
        title = "The Physics of Diffraction: Bending the Light"
        lines = [
            "Diffraction bends light as it passes through tiny openings.",
            "Holographic patterns act as microscopic gratings for light waves.",
            "This bending recreates the original path of the light."
        ]
        self.setup_layout(title, lines)

        # Assets
        slit_asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/slit.svg"
        screen_asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/screen.svg"

        # === Animation for Lecture Line 1 ===
        # Parallel white wave fronts #FFFFFF approach a narrow vertical [Asset: ...slit.svg] slit.
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        # Load and place slit asset
        slit_svg = SVGMobject(slit_asset_path, color=WHITE)
        self.place_at_grid(slit_svg, "C3", scale_factor=1.0) # Issue 44: Scale 1.0
        
        # Incoming wave fronts
        wave_fronts = VGroup(*[
            Line(UP*1.2, DOWN*1.2, color="#FFFFFF", stroke_width=2)
            for _ in range(4)
        ]).arrange(RIGHT, buff=0.4)
        self.place_at_grid(wave_fronts, "C1", scale_factor=1.0)
        
        # Animate waves approaching the slit
        self.add(slit_svg, wave_fronts)
        self.play(
            wave_fronts.animate.shift(RIGHT * 1.8),
            run_time=3,
            rate_func=linear
        )
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # As the waves pass through, they transform into orange circular arcs #FFA500 spreading outward.
        # Holographic patterns act as microscopic gratings for light waves.
        self.play(
            self.lecture[0].animate.set_color(GREY),
            self.lecture[1].animate.set_color("#FFA500")
        )
        
        # Create orange circular arcs centered at the slit
        slit_center = self.grid["C3"]
        arcs = VGroup(*[
            Arc(
                radius=r, 
                start_angle=-PI/3, 
                angle=2*PI/3, 
                arc_center=slit_center, 
                color="#FFA500",
                stroke_width=3
            ) for r in [0.2, 0.6, 1.0, 1.4]
        ])
        
        # Transition from straight waves to arcs
        self.play(
            FadeOut(wave_fronts),
            Create(arcs),
            run_time=2
        )
        
        # Multiple [Asset: ...slit.svg] slits appear (grating)
        # We create a grating using multiple copies of the slit SVG
        grating = VGroup(*[
            SVGMobject(slit_asset_path, color=WHITE).scale(0.3).shift(UP * i * 0.4)
            for i in range(-3, 4)
        ])
        self.place_at_grid(grating, "C3", scale_factor=1.0) # Issue 45: Scale 1.0
        
        # Grating equation
        equation = MathTex("d \\sin(\\theta) = m\\lambda", color="#FFA500", font_size=32)
        self.place_at_grid(equation, "B5") # Issue 43: Position B5
        
        self.play(
            FadeOut(slit_svg),
            FadeIn(grating),
            Write(equation),
            arcs.animate.scale(1.3).shift(RIGHT * 0.3),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Multiple slits appear, creating a pattern of bright and dark spots #FF6347 on a distant [Asset: ...screen.svg] screen.
        self.play(
            self.lecture[1].animate.set_color(GREY),
            self.lecture[2].animate.set_color("#FF6347")
        )
        
        # Distant screen asset at column 6
        screen_svg = SVGMobject(screen_asset_path, color=GREY_B)
        self.place_at_grid(screen_svg, "C6", scale_factor=1.5)
        
        # Interference pattern: Bright spots
        spots = VGroup()
        for i in range(-3, 4):
            # Diminishing intensity for higher order spots
            opacity = 1.0 if i == 0 else (0.7 if abs(i) == 1 else (0.4 if abs(i) == 2 else 0.2))
            dot = Dot(color="#FF6347", radius=0.18).set_opacity(opacity)
            spots.add(dot.shift(UP * i * 0.75))
        self.place_at_grid(spots, "C6", scale_factor=1.0)
        
        # Light rays from grating to spots to visualize the "bending"
        grating_pos = self.grid["C3"]
        rays = VGroup(*[
            Line(grating_pos, spot.get_center(), color="#FF6347", stroke_width=1).set_opacity(0.3)
            for spot in spots
        ])
        
        self.play(
            Create(screen_svg),
            LaggedStart(*[Create(r) for r in rays], lag_ratio=0.1),
            FadeIn(spots, shift=RIGHT*0.2),
            run_time=3
        )
        self.wait(3)
