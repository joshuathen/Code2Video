from manim import *
import numpy as np

# Use the provided TeachingScene base class without modification.
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
        # Data from storyboard
        lecture_lines = [
            "Holographic storage can hold massive amounts of data.",
            "Security holograms on cards prevent counterfeiting via diffraction.",
            "Advanced holography enables high-precision 3D medical imaging."
        ]
        self.setup_layout("Real-World Applications", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Highlight current lecture line
        self.lecture[0].set_color(GOLD)
        
        # Visual representation of a data crystal icon
        crystal_body = Polygon([-0.5, 0, 0], [0, 1, 0], [0.5, 0, 0], [0, -1, 0], color=GOLD, fill_opacity=0.2)
        crystal_lines = VGroup(
            Line([0, 0.5, 0], [0, -0.5, 0], color=GOLD),
            Line([-0.25, 0, 0], [0.25, 0, 0], color=GOLD),
            Line([-0.1, 0.2, 0], [0.1, 0.2, 0], color=GOLD)
        )
        crystal = VGroup(crystal_body, crystal_lines)
        # Grid positioning: Place crystal icon in the top right area
        self.place_at_grid(crystal, "B2", scale_factor=0.7)
        
        storage_label = Text("Terabyte Storage", font_size=20, color=GOLD)
        # Issue 38 Fix: place_in_area for storage_label
        self.place_in_area(storage_label, 'B3', 'C4', scale_factor=0.8)
        
        self.play(FadeIn(crystal), Write(storage_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Reset previous and highlight current line
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(BLUE_C)
        
        # Visual representation of a Credit Card with an Owl hologram
        card = RoundedRectangle(corner_radius=0.1, height=1.5, width=2.5, color=GREY_B, fill_opacity=0.3)
        self.place_at_grid(card, "D2", scale_factor=1.0)
        
        # Owl Hologram using provided SVG asset
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/owl.svg]
        owl = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/owl.svg")
        owl.set_color("#FFD700")
        owl.scale(0.3)
        owl.move_to(card.get_center() + RIGHT*0.6) # Position owl on card
        
        bragg_label = Text("Bragg Law", font_size=20, color=WHITE)
        self.place_at_grid(bragg_label, "D3", scale_factor=1.0)
        
        self.play(FadeIn(card), FadeIn(owl))
        
        # Tilt animation using ValueTracker to simulate diffraction effect
        # We simulate the "wings changing position" by scaling/rotating the owl slightly during the tilt
        tilt_tracker = ValueTracker(0)
        
        def update_owl(m):
            # Simulate diffraction shift by slightly changing scale and rotation as card tilts
            val = tilt_tracker.get_value()
            m.set_opacity(0.7 + 0.3 * np.cos(val * PI))
            m.scale(1.0 + 0.05 * np.sin(val * PI))

        owl.add_updater(update_owl)
        
        # Animation: Tilt card to show diffraction effect
        self.play(
            card.animate.rotate(0.3, axis=RIGHT),
            owl.animate.rotate(0.3, axis=RIGHT),
            tilt_tracker.animate.set_value(1),
            Write(bragg_label),
            run_time=1.5
        )
        self.play(
            card.animate.rotate(-0.6, axis=RIGHT),
            owl.animate.rotate(-0.6, axis=RIGHT),
            tilt_tracker.animate.set_value(-1),
            run_time=2
        )
        self.play(
            card.animate.rotate(0.3, axis=RIGHT),
            owl.animate.rotate(0.3, axis=RIGHT),
            tilt_tracker.animate.set_value(0),
            run_time=1.5
        )
        
        owl.remove_updater(update_owl)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Reset previous and highlight current line
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(RED_A)
        
        # Visual representation of a 3D Brain Scan icon
        brain_base = Circle(radius=0.45, color=RED_A, fill_opacity=0.3)
        brain_l = Circle(radius=0.35, color=RED_A, fill_opacity=0.3).shift(LEFT*0.1)
        brain_r = Circle(radius=0.35, color=RED_A, fill_opacity=0.3).shift(RIGHT*0.1)
        brain_detail = VGroup(
            Line(color=RED_A, stroke_width=2).scale(0.3).move_to(LEFT*0.1+UP*0.1),
            Line(color=RED_A, stroke_width=2).scale(0.3).move_to(RIGHT*0.1+DOWN*0.1)
        )
        
        brain = VGroup(brain_base, brain_l, brain_r, brain_detail)
        self.place_at_grid(brain, "F2", scale_factor=1.0)
        
        mri_label = Text("3D Medical Scan", font_size=20, color=RED_A)
        # Issue 39 Fix: place_in_area for mri_label
        self.place_in_area(mri_label, 'F3', 'F5', scale_factor=0.8)
        
        self.play(FadeIn(brain), Write(mri_label))
        self.wait(2)
