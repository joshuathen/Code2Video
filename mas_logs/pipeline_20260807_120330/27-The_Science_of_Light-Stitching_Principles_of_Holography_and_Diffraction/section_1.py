from manim import *
import numpy as np

# Explicitly define CYAN as it may not be in the global namespace in some Manim CE versions
CYAN = "#00FFFF"

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
        self.setup_layout("Introduction: Beyond the 2D Photo", [
            "Standard photos record only the light's intensity.",
            "They capture a flat, two-dimensional shadow of reality.",
            "Holography records both light intensity and phase.",
            "Phase captures the \"shape\" and \"timing\" of waves.",
            "This creates a \"window with a memory\" of depth."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Line 1: Standard photos record only the light's intensity.
        # Animation: Show a 2D 'Cyber-Cat' photo (flat rectangle) #FFFFFF with a label 'Intensity'.
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        photo_rect = Rectangle(width=2.5, height=1.8, color=WHITE)
        # Fix for Issue 31: Shift visual representation further right.
        self.place_in_area(photo_rect, "B3", "D4")
        
        cat_placeholder = Text("2D Cat", font_size=16, color=WHITE)
        # Fix for Issue 31: Shift visual representation further right.
        self.place_in_area(cat_placeholder, "B3", "D4")
        
        photo_group = VGroup(photo_rect, cat_placeholder)
        
        intensity_label = Text("Intensity", font_size=20, color=WHITE)
        # Fix for Issue 32: Center the label under the image area.
        self.place_in_area(intensity_label, 'E3', 'E4')
        
        self.play(FadeIn(photo_group), Write(intensity_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line 2: They capture a flat, two-dimensional shadow of reality.
        # Animation: Rotate the 2D photo to show it has zero thickness, emphasizing it's a 'flat shadow'.
        self.play(self.lecture[1].animate.set_color(WHITE))
        
        # Rotate around its own center to show flatness
        self.play(
            Rotate(photo_group, angle=75*DEGREES, axis=UP),
            run_time=1.5
        )
        self.wait(0.5)
        self.play(
            Rotate(photo_group, angle=-75*DEGREES, axis=UP),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line 3: Holography records both light intensity and phase.
        # Animation: Show a 3D Cyber-Cat model #00FFFF. Animate light waves (sine waves) hitting it and bouncing off.
        self.play(self.lecture[2].animate.set_color(CYAN))
        
        # Simple 3D cat representation using wireframe-like boxes
        cat_3d = VGroup(
            Rectangle(width=1.2, height=1.2, color=CYAN),
            Rectangle(width=1.2, height=1.2, color=CYAN).shift(0.3*UR)
        )
        cat_3d.add(Line(cat_3d[0].get_corner(UL), cat_3d[1].get_corner(UL), color=CYAN))
        cat_3d.add(Line(cat_3d[0].get_corner(UR), cat_3d[1].get_corner(UR), color=CYAN))
        cat_3d.add(Line(cat_3d[0].get_corner(DL), cat_3d[1].get_corner(DL), color=CYAN))
        cat_3d.add(Line(cat_3d[0].get_corner(DR), cat_3d[1].get_corner(DR), color=CYAN))
        
        # Aligned with the shifted photo position from Issue 31
        self.place_in_area(cat_3d, "B3", "D4")
        
        def get_wave_path(start_pos, end_pos, amplitude=0.1, frequency=10):
            dist = np.linalg.norm(end_pos - start_pos)
            wave = ParametricFunction(
                lambda t: np.array([
                    t,
                    amplitude * np.sin(frequency * t),
                    0
                ]),
                t_range=[0, dist]
            )
            angle = np.arctan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])
            wave.rotate(angle, about_point=ORIGIN)
            wave.move_to((start_pos + end_pos) / 2)
            return wave

        # Move wave start to col 2 to respect B021
        wave_in = get_wave_path(self.grid["A2"], self.grid["B3"])
        wave_in.set_color(CYAN)
        
        self.play(
            FadeOut(photo_group), 
            FadeOut(intensity_label),
            FadeIn(cat_3d)
        )
        self.play(Create(wave_in))
        
        # Reflected waves
        wave_out1 = get_wave_path(self.grid["B3"], self.grid["B6"])
        wave_out2 = get_wave_path(self.grid["D4"], self.grid["D6"])
        wave_out1.set_color(CYAN)
        wave_out2.set_color(CYAN)
        
        self.play(Create(wave_out1), Create(wave_out2))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Line 4: Phase captures the \"shape\" and \"timing\" of waves.
        # Animation: Highlight wave crests on the scattered waves and label them 'Phase' #FFFF00.
        self.play(self.lecture[3].animate.set_color(YELLOW))
        
        phase_label = Text("Phase", font_size=20, color=YELLOW)
        # Fix for Issue 33: Align directly under Intensity label.
        self.place_in_area(phase_label, 'F3', 'F4')
        
        # Extract points for crests
        crests = VGroup()
        for p in [0.2, 0.4, 0.6, 0.8]:
            dot1 = Dot(color=YELLOW, radius=0.06).move_to(wave_out1.point_from_proportion(p))
            dot2 = Dot(color=YELLOW, radius=0.06).move_to(wave_out2.point_from_proportion(p))
            crests.add(dot1, dot2)
            
        self.play(FadeIn(crests), Write(phase_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Line 5: This creates a \"window with a memory\" of depth.
        # Animation: The waves converge onto a glowing 'Holographic Plate' #FFFFFF that stores the 3D wavefront.
        self.play(self.lecture[4].animate.set_color(WHITE))
        
        holo_plate = Rectangle(width=0.3, height=4.0, color=WHITE).set_fill(WHITE, opacity=0.3)
        self.place_in_area(holo_plate, "A6", "F6")
        
        self.play(
            wave_out1.animate.set_color(WHITE),
            wave_out2.animate.set_color(WHITE),
            FadeIn(holo_plate)
        )
        
        # Glow pulse to simulate storage/recording
        self.play(holo_plate.animate.set_fill(opacity=0.8), run_time=0.4)
        self.play(holo_plate.animate.set_fill(opacity=0.3), run_time=0.4)
        
        self.wait(2)
