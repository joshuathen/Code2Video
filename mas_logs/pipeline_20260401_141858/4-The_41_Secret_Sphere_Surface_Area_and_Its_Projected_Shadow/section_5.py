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
        # Setup the layout with section-specific title and lecture lines
        lecture_lines = [
            "Earth intercepts solar energy like a flat circular disc.",
            "This energy spreads across the entire rotating surface area.",
            "Thus, heat distributes at a precise one-to-four ratio."
        ]
        self.setup_layout("Application: Global Energy Balance", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Earth intercepts solar energy like a flat circular disc.
        self.lecture[0].set_color("#FFFF00")
        
        # Create Earth as a blue circle
        earth = Circle(radius=1.2, color="#0000FF", fill_opacity=0.3)
        self.place_in_area(earth, "B3", "E5")
        
        # Create Sunlight Beams (Yellow lines)
        beams = VGroup(*[
            Line(start=LEFT * 1.5, end=ORIGIN, color="#FFFF00", stroke_width=4)
            for _ in range(7)
        ]).arrange(DOWN, buff=0.3)
        # Center beams to line up with Earth's cross-section
        beams.move_to(earth.get_center() + LEFT * 1.5)
        
        # Intercepted Disc (Visualized as a vertical yellow cross-section)
        intercept_disc = Ellipse(width=0.2, height=2.4, color="#FFFF00", fill_opacity=0.9)
        intercept_disc.move_to(earth.get_center())
        
        self.play(FadeIn(earth))
        self.play(Create(beams))
        self.play(FadeIn(intercept_disc))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # This energy spreads across the entire rotating surface area.
        self.lecture[1].set_color("#FFFF00")
        
        # Animate the yellow light spreading to cover the entire blue circle
        self.play(
            earth.animate.set_fill("#FFFF00", opacity=0.5),
            FadeOut(intercept_disc),
            FadeOut(beams),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Thus, heat distributes at a precise one-to-four ratio.
        self.lecture[2].set_color("#FFFF00")
        
        # Label cross-section as 'πr²' (white)
        label_intercepted = Text("πr²", color="#FFFFFF", font_size=24)
        self.place_in_area(label_intercepted, "A4", "A5", scale_factor=0.8)
        
        # Label surface area as '4πr²' (white)
        label_surface = Text("4πr²", color="#FFFFFF", font_size=24)
        self.place_in_area(label_surface, "F4", "F5", scale_factor=0.8)
        
        # Ratio text (Issue 43 fix)
        ratio_text = Text("Ratio: 1 to 4", color="#FFFFFF", font_size=24)
        self.place_in_area(ratio_text, "B1", "B2", scale_factor=0.7)
        
        # Display 'Intensity = 1/4' (yellow) with a white emphasis box (Issue 44 fix)
        intensity_formula = Text("Intensity = 1/4", color="#FFFF00", font_size=28)
        self.place_in_area(intensity_formula, "E1", "E2", scale_factor=0.7)
        box = SurroundingRectangle(intensity_formula, color="#FFFFFF", buff=0.1)
        intensity_group = VGroup(intensity_formula, box)
        
        # Planetary Climate Balance label
        climate_label = Text("Planetary Climate Balance", color="#FFFFFF", font_size=24)
        self.place_in_area(climate_label, "A1", "A3", scale_factor=0.7)
        
        self.play(
            Write(label_intercepted),
            Write(label_surface),
            Write(ratio_text)
        )
        self.play(FadeIn(intensity_group))
        self.play(Write(climate_label))
        
        # Show Earth rotating
        self.play(Rotate(earth, angle=TAU, run_time=3))
        
        self.wait(2)
