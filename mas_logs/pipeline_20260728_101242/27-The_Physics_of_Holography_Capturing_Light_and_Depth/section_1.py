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
        # Define asset paths
        ASSET_FISH = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/fish.svg"
        ASSET_CAT = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/cat.png"

        self.setup_layout("The 2D vs. 3D Paradox", [
            "Photography only records the intensity of light waves.",
            "Holography captures both light intensity and phase information.",
            "Phase provides the depth information missing in 2D photos."
        ])

        # Colors
        COLOR_2D = "#FFFFFF"
        COLOR_3D = "#00FF00"
        HIGHLIGHT = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        # "Photography only records the intensity of light waves."
        self.play(self.lecture[0].animate.set_color(HIGHLIGHT))

        # Create 2D Fish Photo
        photo_rect = Rectangle(width=2.2, height=1.6, color=COLOR_2D, fill_opacity=0.1)
        # Using asset for fish
        fish_2d = SVGMobject(ASSET_FISH).set_color(COLOR_2D)
        # Place fish inside the photo_rect before grouping
        fish_2d.scale(0.3)
        photo_content = VGroup(photo_rect, fish_2d)
        
        photo_label = Text("Amplitude Only", font_size=18, color=COLOR_2D)
        
        self.place_in_area(photo_content, "B1", "C3")
        # Fix Issue 27: Relocate photo_label to D1 and rescale
        self.place_at_grid(photo_label, "D1", scale_factor=0.8)
        
        self.play(FadeIn(photo_content), Write(photo_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Holography captures both light intensity and phase information."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(HIGHLIGHT)
        )

        # Robotic Cat Character
        # Using asset for cat
        cat = ImageMobject(ASSET_CAT)
        # Fix Issue 25: Relocate cat to F2 and rescale
        self.place_at_grid(cat, "F2", scale_factor=0.5)
        
        # Cat enters and looks at photo
        # Adjusted movement path to stay clear of labels
        self.play(cat.animate.move_to(self.grid["E2"])) 
        self.wait(1)

        # 3D holographic fish
        # Use same fish asset but different styling
        fish_3d = SVGMobject(ASSET_FISH).set_color(COLOR_3D).set_opacity(0.6)
        fish_3d.scale(0.4)
        # Add a "glow"
        glow = fish_3d.copy().scale(1.1).set_color(COLOR_3D).set_opacity(0.2)
        hologram_group = VGroup(glow, fish_3d)
        
        hologram_label = Text("Amplitude + Phase", font_size=18, color=COLOR_3D)
        
        self.place_in_area(hologram_group, "B4", "C6")
        # Fix Issue 26: Relocate hologram_label to D6 and rescale
        self.place_at_grid(hologram_label, "D6", scale_factor=0.8)
        
        self.play(FadeIn(hologram_group), Write(hologram_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Phase provides the depth information missing in 2D photos."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(HIGHLIGHT)
        )

        # Robotic cat reaches behind the 3D fish to demonstrate depth.
        # Set z_index to show cat is behind fish
        hologram_group.set_z_index(10)
        cat.set_z_index(5)
        
        # Move cat to the holographic fish position area
        self.play(cat.animate.move_to(self.grid["C5"]))
        
        # Fish "rotates" to show depth - using stretch to avoid 3D axis issues while simulating rotation
        self.play(
            hologram_group.animate.stretch(0.4, 0),
            run_time=1,
            rate_func=smooth
        )
        self.play(
            hologram_group.animate.stretch(2.5, 0),
            run_time=1,
            rate_func=smooth
        )

        # Highlight 'Phase' text with a pulse effect
        # Create a specific pulse for the word 'Phase' if possible, or just the whole label
        # The label is "Amplitude + Phase"
        self.play(
            hologram_label.animate.scale(1.2).set_color(HIGHLIGHT),
            rate_func=there_and_back,
            run_time=1.5
        )
        
        self.wait(2)

        # Reset colors
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
