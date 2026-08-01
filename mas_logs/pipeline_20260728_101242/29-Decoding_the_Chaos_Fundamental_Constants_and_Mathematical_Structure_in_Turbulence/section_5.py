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

class Section5Scene(TeachingScene):
    def construct(self):
        # Title and Lecture Lines
        title_text = "The Dissipation Scale: Where Math Meets Heat"
        lecture_lines = [
            "At the Kolmogorov microscale, viscosity finally takes control.",
            "Kinetic energy is no longer passed down to smaller scales.",
            "Molecular friction converts the remaining motion into thermal heat."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        BLUE_COL = "#ADD8E6"
        HEAT_COL = "#FF4500"
        WHITE_COL = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Highlight lecture line 1
        self.play(self.lecture[0].animate.set_color(BLUE_COL))
        
        # Create a cluster of tiny spinning dots representing the Kolmogorov scale
        dots = VGroup()
        for i in range(12):
            angle = i * (2 * PI / 12)
            # Create a ring of dots to visually represent tiny vortices
            dot = Dot(radius=0.04, color=BLUE_COL)
            dot.shift(0.5 * np.array([np.cos(angle), np.sin(angle), 0]))
            dots.add(dot)
        
        # Position dots in the lower-middle grid area
        self.place_in_area(dots, 'C3', 'E5')
        
        # Add label for the scale
        label_eta = Text("Kolmogorov Scale (eta)", font_size=24, color=WHITE_COL)
        # Fix for Issue 28 & 30: Use place_in_area for better centering and clearance
        self.place_in_area(label_eta, 'B3', 'B5', scale_factor=0.8)
        
        # Show elements
        self.play(FadeIn(dots), FadeIn(label_eta))
        
        # Add persistent rotation to simulate tiny eddies
        dots.add_updater(lambda m, dt: m.rotate(dt * 2))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Highlight lecture line 2
        self.play(self.lecture[1].animate.set_color(HEAT_COL))
        
        # Dots change color to glowing orange-red, indicating the end of the kinetic cascade
        self.play(dots.animate.set_color(HEAT_COL))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Highlight lecture line 3
        self.play(self.lecture[2].animate.set_color(HEAT_COL))
        
        # Create new label for thermal dissipation
        label_dissipation = Text("Thermal Dissipation", font_size=24, color=HEAT_COL)
        # Fix for Issue 29 & 30: Use place_in_area for better centering and clearance
        self.place_in_area(label_dissipation, 'B3', 'B5', scale_factor=0.8)
        
        # Transition: Dots fade away as they "turn into heat", label updates
        self.play(
            FadeOut(dots),
            ReplacementTransform(label_eta, label_dissipation)
        )
        self.wait(3)
