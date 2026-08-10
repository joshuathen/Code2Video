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
        lecture_lines = ["High-dimensional spheres concentrate volume near the surface.", "The core remains essentially empty.", "Imagine an orange: the peel holds the mass."]
        self.setup_layout("The 'Spiky' Phenomenon: Where is the Volume?", lecture_lines)
        
        # Load asset
        orange_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/orange.svg"
        
        # === Animation for Lecture Line 1 ===
        # Draw a 'spike' protruding from the center of a hypercube, utilizing the orange asset.
        spike = SVGMobject(orange_asset, color="#FF4500")
        self.place_at_grid(spike, 'D5', scale_factor=0.6)
        self.play(FadeIn(spike))
        self.lecture[0].set_color("#FF4500")

        # === Animation for Lecture Line 2 ===
        # Show the volume concentrating at the spikes. Use flashing effect on spike.
        self.play(Indicate(spike, color="#00CED1", scale_factor=1.2))
        self.lecture[1].set_color("#00CED1")

        # === Animation for Lecture Line 3 ===
        # Explain why intuition fails in higher dimensions. Overlay text 'Intuition Gap'.
        intuition_gap = Text("Intuition Gap", font_size=36, color="#FFD700")
        self.place_at_grid(intuition_gap, 'E6', scale_factor=0.7)
        
        # Placeholder for the requested visual summary/anchor
        lecture_summary_box = VGroup(
            SVGMobject(orange_asset, color=WHITE, fill_opacity=0.3),
            Text("Intuition Gap", font_size=20, color="#FFD700")
        ).arrange(DOWN)
        self.place_in_area(lecture_summary_box, 'A5', 'C6', scale_factor=0.5)
        
        self.play(Write(intuition_gap), FadeIn(lecture_summary_box))
        self.lecture[2].set_color("#FFD700")
