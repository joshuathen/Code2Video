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
        title_text = "Universal Constants and Real-World Impact"
        lecture_lines = [
            "The Kolmogorov constant provides a universal scale.",
            "This math powers weather models and aircraft design.",
            "Order exists within the heart of chaos."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors for lines and elements
        color_line1 = "#FFD700"  # Golden
        color_line2 = "#87CEEB"  # Sky Blue
        color_line3 = "#FF4500"  # Orange-Red

        # === Animation for Lecture Line 1 ===
        # L: 'The Kolmogorov constant provides a universal scale.'
        # A: Display 'C_K approx 0.5' in golden (#FFD700).
        self.play(self.lecture[0].animate.set_color(color_line1))
        
        ck_text = MathTex(r"C_K \approx 0.5", color=color_line1)
        # Resolved Issue 38: Move to A3-B6 to prevent overlap and visual jump
        self.place_in_area(ck_text, "A3", "B6", scale_factor=1.5)
        
        self.play(Write(ck_text))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # L: 'This math powers weather models and aircraft design.'
        # A: Split screen (within visual area): storm vs airplane wing.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(color_line2)
        )

        # Weather Storm Silhouette (Spiral representing flow)
        storm = VGroup()
        for i in range(1, 6):
            storm.add(Arc(radius=i*0.15, start_angle=0, angle=TAU*0.7, color=BLUE_C, stroke_width=2))
        
        # Resolved Issue 39: Consistent scaling (storm to 0.7)
        self.place_in_area(storm, "D3", "F4", scale_factor=0.7)
        
        storm_label = Text("Weather", font_size=24, color=BLUE_C)
        self.place_at_grid(storm_label, "C3", scale_factor=0.6)

        # Airplane Wing Silhouette (Polygon representing airfoil)
        wing_points = [
            [-1.2, 0, 0], [-0.5, 0.3, 0], [0.8, 0.1, 0], 
            [1.2, 0, 0], [0.8, -0.1, 0], [-0.5, -0.2, 0]
        ]
        wing = Polygon(*wing_points, color=LIGHT_GREY, fill_opacity=0.6, stroke_width=2)
        
        # Resolved Issue 39: Consistent scaling (wing to 0.7)
        self.place_in_area(wing, "D5", "F6", scale_factor=0.7)
        
        wing_label = Text("Aircraft", font_size=24, color=LIGHT_GREY)
        self.place_at_grid(wing_label, "C5", scale_factor=0.6)

        self.play(
            FadeOut(ck_text),
            Create(storm),
            Write(storm_label),
            Create(wing),
            Write(wing_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # L: 'Order exists within the heart of chaos.'
        # A: Overlay -5/3 formula.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(color_line3)
        )

        power_law = MathTex(r"E(k) \propto k^{-5/3}", color=color_line3)
        # Resolved Issue 37: Move to A3-B6 to avoid overlap with 'Weather' and 'Aircraft' labels in row C
        self.place_in_area(power_law, "A3", "B6", scale_factor=1.4)
        
        # Add a semi-transparent background to make the formula pop
        bg_rect = SurroundingRectangle(power_law, color=color_line3, fill_color=BLACK, fill_opacity=0.8, buff=0.2)
        law_group = VGroup(bg_rect, power_law)

        self.play(FadeIn(law_group))
        self.wait(3)
