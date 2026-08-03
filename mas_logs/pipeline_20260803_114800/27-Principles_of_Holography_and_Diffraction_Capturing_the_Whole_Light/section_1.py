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

class Section1Scene(TeachingScene):
    def construct(self):
        title = "Photography vs. Holography: The Missing Dimension"
        lines = [
            "Photography records only light intensity.",
            "It captures a flat, 2D perspective.",
            "Holography records both amplitude and phase.",
            "This recreates the entire light field.",
            "It captures the missing depth dimension."
        ]
        self.setup_layout(title, lines)

        # Assets
        butterfly_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/butterfly.svg"
        camera_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg"

        # Butterfly Mobjects
        # SVGMobjects for assets
        butterfly_2d = SVGMobject(butterfly_asset).set_color("#FFD700")
        butterfly_3d = SVGMobject(butterfly_asset).set_color("#00FFFF")
        
        # Camera Mobject
        camera_icon = SVGMobject(camera_asset).set_color(WHITE).scale(0.5)

        # Light Waves
        wave = FunctionGraph(lambda x: 0.1 * np.sin(5 * x), x_range=[-1.5, 1.5], color=WHITE)
        waves = VGroup(*[wave.copy().shift(UP * i * 0.3) for i in range(-2, 3)])
        
        # Labels
        label_amplitude = Text("Amplitude", font_size=18, color=WHITE)
        label_phase = Text("Phase", font_size=18, color=WHITE)
        label_3d_recon = Text("3D Reconstruction", font_size=20, color="#00FFFF")

        # === Animation for Lecture Line 1 ===
        # Photography records only light intensity.
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        # Issue 25: scale_factor=1.0
        self.place_in_area(butterfly_2d, "B2", "D5", scale_factor=1.0)
        self.play(FadeIn(butterfly_2d))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # It captures a flat, 2D perspective.
        self.play(self.lecture[1].animate.set_color("#FFD700"))
        # Show flatness by a small tilt/squash then back
        self.play(butterfly_2d.animate.stretch(0.1, dim=0), run_time=1)
        self.play(butterfly_2d.animate.stretch(10, dim=0), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Holography records both amplitude and phase.
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        # Issue 25: scale_factor=1.0
        self.place_in_area(butterfly_3d, "B2", "D5", scale_factor=1.0)
        self.place_in_area(waves, "B1", "D3", scale_factor=0.8)
        
        self.play(
            ReplacementTransform(butterfly_2d, butterfly_3d),
            Create(waves)
        )
        
        # Issue 23: labels at B6, C6 with scale 0.8
        self.place_at_grid(label_amplitude, "B6", scale_factor=0.8)
        self.place_at_grid(label_phase, "C6", scale_factor=0.8)
        self.play(Write(label_amplitude), Write(label_phase))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # This recreates the entire light field.
        self.play(self.lecture[3].animate.set_color("#00FFFF"))
        
        # Camera rotation [Asset: camera.svg]
        # We'll place the camera and animate it in an arc around the butterfly area
        center_pos = self.grid["C3"]
        self.place_at_grid(camera_icon, "B1")
        self.play(FadeIn(camera_icon))
        
        # Simulate camera rotation/parallax
        # Rotate camera in arc and rotate butterfly slightly
        self.play(
            MoveAlongPath(camera_icon, Arc(radius=2, start_angle=PI, angle=-PI/2, arc_center=center_pos)),
            butterfly_3d.animate.rotate(PI/6, axis=UP),
            waves.animate.shift(LEFT * 0.2),
            run_time=2
        )
        self.play(
            MoveAlongPath(camera_icon, Arc(radius=2, start_angle=PI/2, angle=PI/2, arc_center=center_pos)),
            butterfly_3d.animate.rotate(-PI/3, axis=UP),
            waves.animate.shift(RIGHT * 0.4),
            run_time=2
        )
        self.play(FadeOut(camera_icon))

        # === Animation for Lecture Line 5 ===
        # It captures the missing depth dimension.
        self.play(self.lecture[4].animate.set_color("#00FFFF"))
        # Issue 24: label_3d_recon in E2-E5 with scale 0.8
        self.place_in_area(label_3d_recon, "E2", "E5", scale_factor=0.8)
        self.play(FadeIn(label_3d_recon))
        
        # Subtle "depth" motion
        self.play(
            butterfly_3d.animate.scale(1.1),
            rate_func=there_and_back,
            run_time=2
        )
        self.wait(2)
